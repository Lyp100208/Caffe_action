#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SingleSliceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const int T = bottom[0]->shape(0);
  const int N = bottom[0]->shape(1);
  const int chn = bottom[0]->count(2);
  CHECK_EQ(N, bottom[1]->count());

  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* real_size = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  CHECK_EQ(top[0]->count(), chn * N * lastN_);

  for (int n = 0; n < N; n++) {
    const int realT = real_size[n];
    CHECK_GE(T, realT) << "real sequence length should be shorter than the one set";

    int t1 = 0;
    for (int t = realT - lastN_; t < realT; t++) {
      caffe_copy(chn, bottom_data + chn * (n + N * t), top_data + chn * (n + N * t1));
      t1++;
    }
  }
}

template <typename Dtype>
void SingleSliceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }

  const int N = bottom[0]->shape(1);
  const int chn = bottom[0]->count(2);

  const Dtype *top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype * real_size = bottom[1]->cpu_data();

  //first all set to zero
  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);

  //then, fill into the lastN_ slices
  for (int n = 0; n < N; n++) {
    const int realT = real_size[n];

    int t1 = 0;
    for (int t = realT - lastN_; t < realT; t++) {
      caffe_copy(chn, top_diff + chn * (n + N * t1), bottom_diff + chn * (n + N * t));
      t1++;
    }
  }

}

INSTANTIATE_LAYER_GPU_FUNCS(SingleSliceLayer);

}  // namespace caffe
