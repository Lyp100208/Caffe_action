#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include  <sys/mman.h>
#include  <sys/types.h>
#include  <sys/stat.h> 
#include  <fcntl.h>

#include "stdint.h"

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/data_layers.hpp"

#include "opencv2/opencv.hpp"


namespace caffe {

template <typename Dtype>
SimpleDataLayer<Dtype>::~SimpleDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void SimpleDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  size_t  channel_[2], height_[2], width_[2];
  const SimpleDataParameter & param = this->layer_param_.simple_data_param();
  crop_size_ = param.crop_size();
  final_size_ = param.final_size();
  crop_center_y_offset_ = param.crop_center_y_offset();
  scale_aug_ = param.scale_aug();
  trans_aug_ = param.trans_aug();
  batch_size_ = param.batch_size();
  channel_[0] = param.channel();
  height_[0] = final_size_;
  width_[0] = final_size_;
  channel_[1] = 1;
  height_[1] = 1;
  width_[1] = 1;
  data_count = height_[0]*width_[0]*channel_[0];
  char buf[1024];
  
  int dataset_size = param.label_source().size();
  CHECK_EQ(dataset_size, param.data_source().size());
  CHECK_EQ(dataset_size, param.image_prefix().size());

  int image_base = 0;
  int label_base = 0;
  for(int d_id = 0; d_id < dataset_size; ++d_id){
    // read label
    FILE *fp = fopen(param.label_source(d_id).c_str(), "r");
    CHECK(fp != NULL) << "file " <<param.label_source(d_id).c_str() << " read error" ;
    int temp_label;
    int image_idx = 0;
    int num = 0;
    int num_people = 0;
    char* result_fgets=fgets(buf,1024,fp);
    if(result_fgets!=buf);
    int len = sscanf(buf, "%d%d", &num, &num_people);
    CHECK_GT(len, 0);
    image_info_.resize(image_base+num);
    while(EOF != fscanf(fp, "%d", &temp_label)){
      CHECK_LT(image_idx, num);
      image_info_[image_idx + image_base].label = temp_label + label_base;
      image_idx++;
    }
    fclose(fp);
    CHECK_EQ(image_idx, num);
    if(len == 1){
      LOG(WARNING) << "Using deprecated label.meta format. Not specifying #people is unsafe";
      num_people = temp_label+1;
    }
    label_base += num_people;
    LOG(INFO) <<"read label done ("<< num_people <<" people)";

    // read list
    LOG(INFO) << "Reading list";
    string image_prefix = param.image_prefix(d_id);
    FILE *fin = fopen(param.data_source(d_id).c_str(), "r");
    char image_fn[1024];
    CHECK(fin != NULL) << "fail to open " << param.data_source(d_id);
    image_idx = 0;
    while(EOF != fscanf(fin, "%s", image_fn)){
      CHECK_LT(image_idx, num);
      image_info_[image_idx + image_base].filename = image_prefix + "/" + string(image_fn);
      //skip the rest of the line
      int num_fscanf=fscanf(fin, "%[^\n]%*c", image_fn);
      if(num_fscanf==0);      
      image_idx++;
    }
    fclose(fin);
    CHECK_EQ(image_idx, num);
    image_base+=num;
    LOG(INFO) <<"read list done (#image, #people: "<<image_info_.size()<<", "<<image_info_[image_info_.size()-1].label+1<<")";
  }
  
  // randomly shuffle data
  LOG(INFO) << "Shuffling data";
  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  ShuffleImages();

  const int thread_id = Caffe::getThreadId();
  const int thread_num = Caffe::getThreadNum();
  current_row_ = image_info_.size() / thread_num * thread_id;

  // getchar();
  // Reshape blobs.
  top[0]->Reshape(batch_size_ / thread_num, channel_[0], height_[0], width_[0]);
  top[1]->Reshape(batch_size_ / thread_num, channel_[1], height_[1], width_[1]);
  LOG(INFO) << "output data size: " << top[0]->num() << ","
	  << top[0]->channels() << "," << top[0]->height() << ","
	  << top[0]->width();
  this->prefetch_data_.Reshape(batch_size_ / thread_num, channel_[0], height_[0], width_[0]);
  this->prefetch_label_.Reshape(batch_size_ / thread_num, channel_[1], height_[1], width_[1]);
}

template <typename Dtype>
void SimpleDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(image_info_.begin(), image_info_.end(), prefetch_rng);
}

template <typename Dtype>
void SimpleDataLayer<Dtype>::InternalThreadEntry(){
//  LOG(INFO) << "current_row_: " << current_row_;
  Dtype *label_ptr_ = this->prefetch_label_.mutable_cpu_data();
  Dtype *data_ptr_ = this->prefetch_data_.mutable_cpu_data();
  for (int i = 0; i < batch_size_ / Caffe::getThreadNum(); ++i){
    crop(image_info_[current_row_].filename, data_ptr_ + i*data_count);
    label_ptr_[i] = image_info_[current_row_].label;
    //LOG(INFO) << "name: " << image_info_[current_row_].filename;
    //LOG(INFO) << "index: "<<current_row_ << " label: "<<label_ptr_[i];
    //getchar(); 
    current_row_ ++;
    if(current_row_ >= image_info_.size()){
      current_row_ = 0;
      // LOG(INFO) << "shuffling";
      ShuffleImages();
    }
  }
}

template <typename Dtype>
void SimpleDataLayer<Dtype>::crop(string path,Dtype* data_out){
  cv::Mat img = cv::imread(path);
  float scale_diff = ((float)(rand()%1000)/500 - 1)*scale_aug_;
  //cout << "scale_diff " << scale_diff << endl;
  float crop_size_aug = (float)crop_size_*(1+scale_diff);
  float trans_diff_x = ((float)(rand()%1000)/500 - 1)*trans_aug_;
  float trans_diff_y = ((float)(rand()%1000)/500 - 1)*trans_aug_;
  
  //cout << "trans_diff_x " << trans_diff_x << endl;
  cv::Point2f center((float)img.cols/2*(1+trans_diff_x), 
    (float)(img.rows/2 + crop_center_y_offset_)*(1+trans_diff_y));
  
  //clip
  if(center.x < crop_size_aug/2) crop_size_aug = center.x*2-0.5;
  if(center.y < crop_size_aug/2) crop_size_aug = center.y*2-0.5;
  if(center.x + crop_size_aug/2 >= img.cols) crop_size_aug = (img.cols-center.x)*2 - 0.5;
  if(center.y + crop_size_aug/2 >= img.rows) crop_size_aug = (img.rows-center.y)*2 - 0.5;
  
  cv::Rect rect(center.x-crop_size_aug/2, center.y-crop_size_aug/2, crop_size_aug, crop_size_aug);
  cv::Mat cropped = img(rect);
  cv::Mat result;
  cv::resize(cropped, result, cv::Size(final_size_, final_size_));
  
  //cv::imwrite("1.jpg", result); 
  uchar * result_ptr = result.data;
  int l = 0;
  int height_mul_width = result.rows*result.cols;
  for(int i=0;i<result.rows;i++)
    for(int j=0;j<result.cols;j++)
      for(int k=0;k<result.channels();k++)
        data_out[k*height_mul_width + i*result.cols + j] = ((float)result_ptr[l++])*3.2/255 - 1.6;
  /*
  cv::Mat temp(result.rows,result.cols,CV_32FC1);
  temp.data=(uchar*)data_out;
  temp.convertTo(temp, CV_8UC1);
  cv::imwrite("1.jpg",temp);
  */
}

INSTANTIATE_CLASS(SimpleDataLayer);
REGISTER_LAYER_CLASS(SimpleData);

}  // namespace caffe
