#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PoolingNumLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PoolingNumParameter pool_param = this->layer_param_.pooling_num_param();
  kernel_size_ = pool_param.kernel_size();
  num_ = bottom[0]->num();
  CHECK((num_ % kernel_size_) == 0) << "mod(num, kernel_size)) != 0";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
}

template <typename Dtype>
void PoolingNumLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
  //    << "corresponding to (num, channels, height, width)";
  CHECK(bottom[0]->num() == num_);

  pooled_num_ = num_ / kernel_size_;
  top[0]->Reshape(pooled_num_, channels_, 1, 1);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }

  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_num_param().pool() ==
      PoolingNumParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(pooled_num_, channels_, 1, 1);
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_num_param().pool() ==
      PoolingNumParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(pooled_num_, channels_, 1, 1);
  }
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void PoolingNumLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LOG(FATAL) << "PoolingNumLayer<Dtype>::Forward_cpu() is not implemented yet";
}

template <typename Dtype>
void PoolingNumLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
  LOG(FATAL) << "PoolingNumLayer<Dtype>::Backward_cpu() is not implemented yet";
}


#ifdef CPU_ONLY
STUB_GPU(PoolingNumLayer);
#endif

INSTANTIATE_CLASS(PoolingNumLayer);
REGISTER_LAYER_CLASS(PoolingNum);

}  // namespace caffe
