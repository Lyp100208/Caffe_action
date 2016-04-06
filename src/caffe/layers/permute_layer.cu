#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void PermuteForward(const int nthreads,
    const Dtype* const bottom_data, const int T, const int batch_size,
    const int dim_fea, Dtype* const top_data) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    //calculate i,t,n in top bottom (T*N * dim_fea)
    const int i = index % dim_fea;
    const int n = (index / dim_fea) % batch_size;
    const int t = index / dim_fea / batch_size;

    //map to index in bottom blob (N*T x dim_fea)
    const int index_bot = dim_fea * (T * n + t) + i;

    top_data[index] = bottom_data[index_bot];
  }
}

template <typename Dtype>
__global__ void PermuteBackward(const int nthreads, const Dtype* const top_diff,
    const int T, const int batch_size, const int dim_fea, Dtype* const bottom_diff) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    //calculate i,t,n in bottom blob (N*T x dim_fea)
    const int i = index % dim_fea;
    const int t = (index / dim_fea) % T;
    const int n = (index / dim_fea) / T;

    //map to index in top bottom (T*N * dim_fea)
    const int index_top = dim_fea * (batch_size * t + n) + i;
    bottom_diff[index] = top_diff[index_top];
  }
}

template <typename Dtype>
void PermuteLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  const Dtype* const bottom_data = bottom[0]->gpu_data();
  Dtype* const top_data = top[0]->mutable_gpu_data();

  PermuteForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, bottom_data, num_seg_, batch_size_, dim_fea_, top_data);

  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void PermuteLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }

  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();

  PermuteBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, top_diff, num_seg_, batch_size_, dim_fea_, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(PermuteLayer);

}  // namespace caffe
