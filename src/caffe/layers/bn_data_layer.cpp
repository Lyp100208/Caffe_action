#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BNDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
    const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  // extract param
  var_eps_ = this->layer_param_.bn_param().var_eps();
  decay_ = this->layer_param_.bn_param().decay();
  moving_average_ = this->layer_param_.bn_param().moving_average();
  
  // reshape blob
  top[0]->Reshape(num_, channels_, height_, width_);
  x_norm_.Reshape(num_, channels_, height_, width_);
  x_std_.Reshape(1, channels_, 1, 1);
  
  // statistic
  spatial_statistic_.Reshape(num_, channels_, 1, 1);
  batch_statistic_.Reshape(1, channels_, 1, 1);
  
  // buffer blob
  buffer_blob_.Reshape(num_, channels_, height_, width_);
  
  // fill spatial multiplier
  spatial_sum_multiplier_.Reshape(1, 1, height_, width_);
  Dtype* spatial_multiplier_data = spatial_sum_multiplier_.mutable_cpu_data();
  caffe_set(spatial_sum_multiplier_.count(), Dtype(1), spatial_multiplier_data);
  // fill batch multiplier
  batch_sum_multiplier_.Reshape(num_, 1, 1, 1);
  Dtype* batch_multiplier_data = batch_sum_multiplier_.mutable_cpu_data();
  caffe_set(batch_sum_multiplier_.count(), Dtype(1), batch_multiplier_data);
  
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(4);
    
    // fill scale with scale_filler
    this->blobs_[0].reset(new Blob<Dtype>(1, channels_, 1, 1));
    if (Caffe::getThreadId() == 0) {
      shared_ptr<Filler<Dtype> > scale_filler(GetFiller<Dtype>(
          this->layer_param_.bn_param().scale_filler()));
      scale_filler->Fill(this->blobs_[0].get());
    }
    Dtype* scale = this->blobs_[0]->mutable_cpu_data(); 
    MPI_Bcast(scale, this->blobs_[0]->count(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    LOG(INFO) << "scale: " << scale[0] << " " << scale[1] << " " << scale[3];
   
    // fill shift with shift_filler
    this->blobs_[1].reset(new Blob<Dtype>(1, channels_, 1, 1));
    if (Caffe::getThreadId() == 0) {
      shared_ptr<Filler<Dtype> > shift_filler(GetFiller<Dtype>(
          this->layer_param_.bn_param().shift_filler()));
      shift_filler->Fill(this->blobs_[1].get());
    }
    Dtype* shift = this->blobs_[1]->mutable_cpu_data(); 
    MPI_Bcast(shift, this->blobs_[1]->count(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    LOG(INFO) << "shift: " << shift[0] << " " << shift[1] << " " << shift[3];
   
    // history mean
    this->blobs_[2].reset(new Blob<Dtype>(1, channels_, 1, 1));
    if (Caffe::getThreadId() == 0) {
      caffe_set(channels_, Dtype(0), this->blobs_[2]->mutable_cpu_data());
    }
    Dtype* mean = this->blobs_[2]->mutable_cpu_data(); 
    MPI_Bcast(mean, this->blobs_[2]->count(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    LOG(INFO) << "mean: " << mean[0] << " " << mean[1] << " " << mean[3];

    // history variance
    this->blobs_[3].reset(new Blob<Dtype>(1, channels_, 1, 1));
    if (Caffe::getThreadId() == 0) {
      caffe_set(channels_, Dtype(0), this->blobs_[3]->mutable_cpu_data());
    }
    Dtype* variance = this->blobs_[3]->mutable_cpu_data(); 
    MPI_Bcast(variance, this->blobs_[3]->count(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    LOG(INFO) << "variance: " << variance[0] << " " << variance[1] << " " << variance[3];
   
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void BNDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void BNDataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(BNDataLayer);
#endif

INSTANTIATE_CLASS(BNDataLayer);
REGISTER_LAYER_CLASS(BNData);
}  // namespace caffe
