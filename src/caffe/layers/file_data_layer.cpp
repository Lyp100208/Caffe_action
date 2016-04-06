#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "stdint.h"

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
FileDataLayer<Dtype>::~FileDataLayer<Dtype>() { }

template <typename Dtype>
void FileDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const VideoDataParameter &para = this->layer_param_.video_data_param();
	const string &root_folder = para.root_folder();
	const string &source = para.source();
	const bool shuffle = para.shuffle();
	const int num_views = para.num_views();
	const int tot_views = para.tot_views();
	CHECK_GE(tot_views, num_views);
	CHECK_GT(num_views, 0) << "num_views need to be set";
	for (int c = 0; c < para.view_ids_size(); ++c) {
		view_ids_.push_back(para.view_ids(c));
	}
	if (view_ids_.empty()) {
		for (int k = 0; k < num_views; k++)
			view_ids_.push_back(k);
	} else {
		CHECK_EQ(num_views, int(view_ids_.size())) << "view_id and num_views conflict";
	}

	const int seq_length = para.seq_length();
	const int num_segments = para.num_segments();
	if (this->phase_ == TRAIN) {
		CHECK_EQ(1, num_views) << "multi-view for training is not supported";
		CHECK_EQ(1, num_segments);
	}
	
	//read list
	LOG(INFO) << "Opening file: " << source;
	std:: ifstream infile(source.c_str());
	string filename;
	int length, label;
	while (infile >> filename >> length >> label){
		lines_.push_back(std::make_pair(filename,label));
	}
	LOG(INFO) << "A total of " << lines_.size() << " videos.";

	const unsigned int prefectch_rng_seed = caffe_rng_rand();
	prefetch_rng_.reset(new Caffe::RNG(prefectch_rng_seed));
	if (shuffle) {
		ShuffleVideos();
	}

	//get line_id_ and batch_size
	const int thread_id = Caffe::getThreadId();
	const int thread_num = Caffe::getThreadNum();
	int batch_size = para.batch_size();
	CHECK_EQ(0, batch_size % thread_num) << ", batchsize = " << batch_size << ", thread_num = " << thread_num;
	batch_size /= thread_num;
	lines_id_ = lines_.size() / thread_num * thread_id;
	
	//Infer top blob sizes
	char path[260];
	sprintf(path, "%s/%s_wholevideo_view%d.bin", root_folder.c_str(), lines_[lines_id_].first.c_str(), view_ids_.empty() ? 0 : view_ids_[0]);
	vector<int> shape;
	LoadBinFile(path, shape);
//	const int T = shape[0];
	chn_ = shape[1];
	CHECK_EQ(1, shape[2]);
	CHECK_EQ(1, shape[3]);
	LOG(INFO) << "shape = [" << shape[0] << ", " << shape[1] << ", " << shape[2] << ", " << shape[3] << "]";

	vector<int> shape_top(4);
	if (para.do_squeeze() && seq_length == 1) {
		shape_top[0] = num_views * num_segments * batch_size;
		shape_top[1] = chn_;
		shape_top[2] = 1;
		shape_top[3] = 1;
	} else {
		shape_top[0] = seq_length;
		shape_top[1] = num_views * num_segments * batch_size;
		shape_top[2] = chn_;
		shape_top[3] = 1;
	}
	LOG(INFO) << "shape_top = [" << shape_top[0] << ", " << shape_top[1] << ", " << shape_top[2] << ", " << shape_top[3] << "]";

	top[0]->Reshape(shape_top); //data
	if (top.size() > 1) {
		shape_top[2] = 1;
		top[1]->Reshape(shape_top); //label
	}
	
	if (top.size() > 2) {
		shape_top[0] = 1;
		top[2]->Reshape(shape_top); //real sequence size
	}
}


template <typename Dtype>
void FileDataLayer<Dtype>::ShuffleVideos(){
	caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
	shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

template <typename Dtype>
void FileDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const VideoDataParameter &para = this->layer_param_.video_data_param();
	const string &root_folder = para.root_folder();
	const bool shuffle = para.shuffle();
	const int num_views = para.num_views();
	const int tot_views = para.tot_views();
	const int seq_length = para.seq_length();
	const int stride_seq = para.stride_seq();
	const int num_segments = para.num_segments();
	const bool do_rolling = para.do_rolling();
	const bool is_train = (this->phase_ == TRAIN);

	const int thread_num = Caffe::getThreadNum();
	int batch_size = para.batch_size() / thread_num;
	const int lines_size = int(lines_.size());

	Dtype *pdata = top[0]->mutable_cpu_data();
	Dtype *plabel = (top.size() > 1) ? top[1]->mutable_cpu_data() : NULL;
	Dtype *psize_seq = (top.size() > 2) ? top[2]->mutable_cpu_data() : NULL;
	if (top.size() > 1)
		caffe_set(top[1]->count(), Dtype(-1), top[1]->mutable_cpu_data());

	char path[260];
	vector<vector<Dtype> > data(num_views);
	for (int b = 0; b < batch_size; b++)
	{
		const int id_vid = (lines_id_ + b) % lines_size;
		const string &name_vid = lines_[id_vid].first;
		const Dtype label = lines_[id_vid].second;

		//randomly choose view_id for training stage
		caffe::rng_t* rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
		if (is_train)
			view_ids_[0] = (*rng)() % tot_views;

		//load feature data for needed views
		vector<int> s, s0;
		for (int v = 0; v < num_views; v++) {
			sprintf(path, "%s/%s_wholevideo_view%d.bin", root_folder.c_str(), name_vid.c_str(), view_ids_[v]);
			LoadBinFile(path, data[v], s);
			if (0 == v)
				s0 = s;
			else {
				CHECK(s0 == s);
			}
		}
		const int len_fea_array = s0[0]; //different from len_vid in list file

		//decide start id for each segment
		vector<int> starts(num_segments, -1);
		vector<int> size_seq(num_segments, seq_length);
		SampleSegments(len_fea_array, seq_length, stride_seq, is_train, do_rolling, num_segments, starts, size_seq, rng);

		//fill data and label
		for (int t = 0; t < seq_length; t++) {
			Dtype *pd = pdata + chn_ * num_views * num_segments * (batch_size * t + b); 
			Dtype *pl = (plabel == NULL) ? NULL : plabel + num_views * num_segments * (batch_size * t + b);
			for (int s = 0; s < num_segments; s++) {
				if (t >= size_seq[s])
					continue;

				int pos = starts[s] + stride_seq * t;
				if (do_rolling)
					pos = pos % len_fea_array;
				else
					CHECK_LT(pos, len_fea_array);
				for (int v = 0; v < num_views; v++) {
					caffe_copy(chn_, &(data[v][chn_ * pos]), pd);
					if (t == size_seq[s]-1) {
						if (pl != NULL)
							*pl = label;
						if (do_rolling)
							CHECK_EQ(t, seq_length-1);
						else
							CHECK(pos + stride_seq >= len_fea_array || t==seq_length-1);
					}
					pd += chn_;
					if (pl != NULL)
						pl += 1;
				}
			}
		}
		if (psize_seq != NULL) {
			//fill length
			Dtype *plen = psize_seq + num_views * num_segments * b;
			for (int s = 0; s < num_segments; s++) {
				for (int v = 0; v < num_views; v++) {
					*plen = size_seq[s];
					plen++;
				}
			}
		}
	}

	//next iteration
	lines_id_ += batch_size;
	if (lines_id_ >= lines_size) {
		DLOG(INFO) << "Restarting data prefetching from start.";
		lines_id_ = 0;
		if(shuffle){
			ShuffleVideos();
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(FileDataLayer, Forward);
#endif

INSTANTIATE_CLASS(FileDataLayer);
REGISTER_LAYER_CLASS(FileData);

}  // namespace caffe
