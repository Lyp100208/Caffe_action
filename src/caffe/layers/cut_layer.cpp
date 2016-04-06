#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void CutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  fold_ = this->layer_param_.cut_param().fold();
  start_ = this->layer_param_.cut_param().start();
  end_ = this->layer_param_.cut_param().end();

  CHECK_GE(fold_, end_) << "end index should not exceed fold number";
  CHECK_GE(end_, start_) << "start index should not larger than end index";

  CHECK_EQ(0, bottom[0]->num() % fold_);
}

template <typename Dtype>
void CutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  vector<int> shape = bottom[0]->shape();
  CHECK_EQ(bottom[0]->num(), shape[0]);
  shape[0] = shape[0] / fold_ * (end_ - start_ + 1);
  top[0]->Reshape(shape);
}

template <typename Dtype>
void CutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LOG(FATAL) << "not implemented";
}

template <typename Dtype>
void CutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL) << "not implemented";
}


#ifdef CPU_ONLY
STUB_GPU(CutLayer);
#endif

INSTANTIATE_CLASS(CutLayer);
REGISTER_LAYER_CLASS(Cut);

}  // namespace caffe
