#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/benchmark.hpp"
#include <omp.h>

#ifdef USE_MPI
#include "mpi.h"
#include <boost/filesystem.hpp>
using namespace boost::filesystem;
#endif

namespace caffe{
template <typename Dtype>
VideoDataRNNLayer<Dtype>:: ~VideoDataRNNLayer<Dtype>(){
	this->JoinPrefetchThread();
}

template <typename Dtype>
void VideoDataRNNLayer<Dtype>:: DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const VideoDataParameter para = this->layer_param_.video_data_param();

	const int new_height  = para.new_height();
	const int new_width  = para.new_width();
	const int new_length  = para.new_length();
	const int num_segments = para.num_segments(); //num_segments will be used in MIL exps
	const int seq_length = para.seq_length();
	const string suffix = para.suffix();
	const string& source = para.source();
	string root_folder = para.root_folder();
	const bool flow_is_color = para.flow_is_color();

	LOG(INFO) << "Opening file: " << source;
	std:: ifstream infile(source.c_str());
	string filename;
	int label;
	int length;
	while (infile >> filename >> length >> label){
		lines_.push_back(std::make_pair(filename,label));
		lines_duration_.push_back(length);
	}
	if (para.shuffle()){
		const unsigned int prefectch_rng_seed = caffe_rng_rand();
		prefetch_rng_1_.reset(new Caffe::RNG(prefectch_rng_seed));
		prefetch_rng_2_.reset(new Caffe::RNG(prefectch_rng_seed));
		ShuffleVideos();
	}

	LOG(INFO) << "A total of " << lines_.size() << " videos.";
	//lines_id_ = 0;
	const int thread_id = Caffe::getThreadId();
	const int thread_num = Caffe::getThreadNum();
	lines_id_ = lines_.size() / thread_num * thread_id;

	Datum datum;
	const unsigned int frame_prefectch_rng_seed = caffe_rng_rand();
	frame_prefetch_rng_.reset(new Caffe::RNG(frame_prefectch_rng_seed));

	const int len_vid = int(lines_duration_[lines_id_]);
	CHECK_GE(len_vid, new_length);

    	vector<int> offsets(1,0);
	if (para.modality() == VideoDataParameter_Modality_FLOW)
		CHECK(ReadSegmentFlowToDatum(root_folder + lines_[lines_id_].first, lines_[lines_id_].second, offsets, new_height, new_width, new_length, &datum, flow_is_color));
	else
		CHECK(ReadSegmentRGBToDatum(root_folder + lines_[lines_id_].first, lines_[lines_id_].second, offsets, new_height, new_width, new_length, &datum, true, 0, suffix));
	const int crop_size = this->layer_param_.transform_param().crop_size();
	int batch_size = para.batch_size();
	CHECK(batch_size % thread_num == 0) << "batch_size % thread_num != 0";
	batch_size /= thread_num;

	const int NxT = num_segments * batch_size * seq_length;
	if (crop_size > 0){
		top[0]->Reshape(NxT, datum.channels(), crop_size, crop_size);
		this->prefetch_data_.Reshape(NxT, datum.channels(), crop_size, crop_size);
	} else {
		top[0]->Reshape(NxT, datum.channels(), datum.height(), datum.width());
		this->prefetch_data_.Reshape(NxT, datum.channels(), datum.height(), datum.width());
	}

	vector<int> shape_lab(3, 1); //T * N * 1
	shape_lab[0] = seq_length;
	shape_lab[1] = num_segments * batch_size;
	top[1]->Reshape(shape_lab);
	this->label_blob_.Reshape(shape_lab);
	this->prefetch_labels_.push_back(&this->label_blob_);

	//shape_lab[0] = 1;
	//top[2]->Reshape(shape_lab); //1 * N * 1: record the real sequence length
	top[2]->Reshape(shape_lab); //T * N * 1: marker
	this->marker_blob_.Reshape(shape_lab);
	this->prefetch_labels_.push_back(&this->marker_blob_);
}

template <typename Dtype>
void VideoDataRNNLayer<Dtype>::ShuffleVideos(){
	caffe::rng_t* prefetch_rng1 = static_cast<caffe::rng_t*>(prefetch_rng_1_->generator());
	caffe::rng_t* prefetch_rng2 = static_cast<caffe::rng_t*>(prefetch_rng_2_->generator());
	shuffle(lines_.begin(), lines_.end(), prefetch_rng1);
	shuffle(lines_duration_.begin(), lines_duration_.end(),prefetch_rng2);
}

template <typename Dtype>
void VideoDataRNNLayer<Dtype>::InternalThreadEntry(){
	CHECK(this->prefetch_data_.count());

	const VideoDataParameter &para = this->layer_param_.video_data_param();
	const string &root_folder = para.root_folder();

	const int new_width = para.new_width();
	const int new_height = para.new_height();
	const int new_length = para.new_length();

	const bool shuffle = para.shuffle();
	const int seq_length = para.seq_length();
	const int stride_seq = para.stride_seq();
	const int num_segments = para.num_segments();
	const bool do_rolling = para.do_rolling();
	const bool is_train = (this->phase_ == TRAIN);
	const bool flow_is_color = para.flow_is_color();
	const bool eval_last = para.eval_last();
	const string suffix = para.suffix();

	const int thread_num = Caffe::getThreadNum();
	int batch_size = para.batch_size() / thread_num;
	const int lines_size = int(lines_.size());

	const int N = num_segments * batch_size;
	const int T = seq_length;
	CHECK_EQ(N * T, this->prefetch_data_.num());
	const int chn = this->prefetch_data_.count(1);

	Dtype* pdata = this->prefetch_data_.mutable_cpu_data();
	Dtype* plabel = this->prefetch_labels_[0]->mutable_cpu_data();
	caffe_set(T * N, Dtype(-1), plabel);
	Dtype* pmarker = this->prefetch_labels_[1]->mutable_cpu_data();

	omp_set_num_threads(sysconf(_SC_NPROCESSORS_ONLN) / thread_num);
#pragma omp parallel for
	for (int b = 0; b < batch_size; b++)
	{
		const int id_vid = (lines_id_ + b) % lines_size;
		const Dtype label = lines_[id_vid].second;
		const int len_vid = lines_duration_[id_vid];
		CHECK_GE(len_vid, new_length + stride_seq - 1);
		const int len_fea_arr = len_vid - new_length + 1;

		//decide start id for each segment in the same video
		vector<int> starts(num_segments, -1);
		vector<int> realsize(num_segments, seq_length);
		caffe::rng_t* rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
		SampleSegments(len_fea_arr, seq_length, stride_seq, is_train, do_rolling, num_segments, starts, realsize, rng);

		//sequence-by-sequence
		Datum datum;
		Blob<Dtype> transformed_data_loc;
		for (int s = 0; s < num_segments; s++) {
			const int s0 = starts[s];
			CHECK_GE(s0, 0);
			const int len = realsize[s];
			CHECK_GT(len, 0);

			//node-by-node within a sequence
			vector<int> offsets(1);
			const bool do_mirror = this->layer_param_.transform_param().mirror()
							 && ((*rng)() % 2);
			const int preset_mirror = do_mirror ? 1 : 0;
//			LOG(INFO) << "preset_mirror = " << preset_mirror;
			for (int t = 0; t < len; t++) {
				if (do_rolling)
					offsets[0] = (s0 + stride_seq * t) % len_fea_arr;
				else
					offsets[0] = s0 + stride_seq * t;

				//read images
				if (para.modality() == VideoDataParameter_Modality_FLOW)
					CHECK(ReadSegmentFlowToDatum(root_folder + lines_[id_vid].first, lines_[id_vid].second,
						 offsets, new_height, new_width, new_length, &datum, flow_is_color));
				else
					CHECK(ReadSegmentRGBToDatum(root_folder + lines_[id_vid].first, lines_[id_vid].second,
						 offsets, new_height, new_width, new_length, &datum, true, 0, suffix));

				//do transformation
				vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
				transformed_data_loc.Reshape(top_shape);
				transformed_data_loc.set_cpu_data(pdata + chn * (s + num_segments * (b + batch_size * t)));
				const int chn_flow_single = flow_is_color ? 3 : 1;
				this->data_transformer_->Transform(datum, &(transformed_data_loc), chn_flow_single, preset_mirror);

				//copy to data, be careful. The real size of top[0] is: T * batch_size * num_segments * (C * H * W)
				CHECK_EQ(transformed_data_loc.count(), chn);
				if(eval_last == false) {
					//label
					plabel[s + num_segments * (b + batch_size * t)] = label; //last element
				}
				if(t == 0) {
					pmarker[s + num_segments * (b + batch_size * t)] = -len;
				} else {
					pmarker[s + num_segments * (b + batch_size * t)] = 1;
				}
			}
			if(eval_last == true) {
				//label
				plabel[s + num_segments * (b + batch_size * (len-1))] = label; //last element
			}
			//sequence real length
			//preal_size[s + num_segments * b] = len;
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

INSTANTIATE_CLASS(VideoDataRNNLayer);
REGISTER_LAYER_CLASS(VideoDataRNN);
}
