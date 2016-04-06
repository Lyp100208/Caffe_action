#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
void PermuteLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.permute_param().axis());
  const int num_tot = bottom[0]->count(0, axis_);
  
  num_seg_ = this->layer_param_.permute_param().num_seg(); //T
  CHECK_EQ(num_tot % num_seg_, 0) << "number dimention should be times of num_seg_ " << "num_tot = " << num_tot << " num_seg_ = " << num_seg_;
  batch_size_ =  num_tot / num_seg_; //N
  dim_fea_ = bottom[0]->count(axis_);
}

template <typename Dtype>
void PermuteLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> shape(3);
  shape[0] = num_seg_; //T
  shape[1] = batch_size_; //N
  shape[2] = dim_fea_;

  top[0]->Reshape(shape);
}

template <typename Dtype>
void PermuteLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LOG(FATAL) << "Forward_cpu is not implemented yet";
}

template <typename Dtype>
void PermuteLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL) << "Backward_cpu is not implemented yet";
}

#ifdef CPU_ONLY
STUB_GPU(PermuteLayer);
#endif

INSTANTIATE_CLASS(PermuteLayer);
REGISTER_LAYER_CLASS(Permute);

}  // namespace caffe
