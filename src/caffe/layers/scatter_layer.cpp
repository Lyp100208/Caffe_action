#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
template <typename Dtype>
void ScatterLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void ScatterLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // The top shape will be the bottom shape with the flattened axes dropped,
  for (int i = 0; i < bottom.size(); ++i) {
    vector<int> shape = bottom[i]->shape();
    CHECK_GE(shape.size() , 0);
    shape[0] = shape[0] / Caffe::getThreadNum();
    top[i]->Reshape( shape );
  }
}

template <typename Dtype>
void ScatterLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void ScatterLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(ScatterLayer);
#endif

INSTANTIATE_CLASS(ScatterLayer);
REGISTER_LAYER_CLASS(Scatter);
}  // namespace caffe
