#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void CutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  caffe_copy(bottom[0]->count() / fold_ * (end_ - start_ + 1),
    bottom[0]->gpu_data() + bottom[0]->count() / fold_ * start_, 
    top[0]->mutable_gpu_data());
}

template <typename Dtype>
void CutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }

  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());
  caffe_copy(top[0]->count(), top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff() + bottom[0]->count() / fold_ * start_);
}


INSTANTIATE_LAYER_GPU_FUNCS(CutLayer);

}  // namespace caffe
