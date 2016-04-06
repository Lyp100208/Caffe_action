#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SingleSliceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(2, bottom.size()) << "SingleSlice expect a data and size input blob";
  CHECK_EQ(1, top.size());

  const SliceParameter& slice_param = this->layer_param_.slice_param();
  lastN_ = slice_param.last_n();
//  slice_axis_ = 0;
  CHECK_EQ(bottom[1]->count(), bottom[0]->shape(1));
}

template <typename Dtype>
void SingleSliceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape = bottom[0]->shape(); // T * N * chn
  top_shape[0] = lastN_;

  if (this->layer_param_.slice_param().do_squeeze() && lastN_ == 1)
    top_shape.erase(top_shape.begin()); //do squeeze

  top[0]->Reshape(top_shape);
//  num_slices_ = bottom[0]->count(0, slice_axis_);
//  slice_size_ = bottom[0]->count(slice_axis_ + 1);
}

template <typename Dtype>
void SingleSliceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int T = bottom[0]->shape(0);
  const int N = bottom[0]->shape(1);
  const int chn = bottom[0]->count(2);

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* real_size = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  for (int n = 0; n < N; n++) {
    const int realT = real_size[n];
    CHECK_GE(T, realT);

    int t1 = 0;
    for (int t = realT - lastN_; t < realT; t++) {
      caffe_copy(chn, bottom_data + chn * (n + N * t), top_data + chn * (n + N * t1));
      t1++;
    }
  }
}

template <typename Dtype>
void SingleSliceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  CHECK(!propagate_down[0]) << "Backward is not implemented yet";
}

#ifdef CPU_ONLY
STUB_GPU(SingleSliceLayer);
#endif

INSTANTIATE_CLASS(SingleSliceLayer);
REGISTER_LAYER_CLASS(SingleSlice);

}  // namespace caffe
