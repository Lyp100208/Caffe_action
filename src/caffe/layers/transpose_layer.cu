#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void TransposeDimensionWithLastDimGPU(const int nthreads,
    const Dtype* input_data, Dtype* output_data, const int count_dim_1, const int count_after_dim_1, 
    const int count_dim_2, const int count_after_dim_2, const int new_count_dim_1,
    const int new_count_after_dim_1, const int new_count_dim_2) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int index_pre=index/count_dim_1;
    int index_dim_1=(index%count_dim_1)/count_after_dim_1;
    int index_insied=(index%count_after_dim_1)/count_dim_2;
    int index_dim_2=(index%count_dim_2)/count_after_dim_2;
    int index_after=index%count_after_dim_2;
    int new_index=index_pre*new_count_dim_1+
                  index_dim_2*new_count_after_dim_1+
                  index_insied*new_count_dim_2+
                  index_dim_1*count_after_dim_2+
                  index_after;
    output_data[new_index]=input_data[index];
  }
}

template <typename Dtype>
__global__ void TransposeDimensionWithoutLastDimGPU(const int nthreads,
    const Dtype* input_data, Dtype* output_data, const int count_dim_1, const int count_after_dim_1, 
    const int count_dim_2, const int new_count_dim_1,
    const int new_count_after_dim_1, const int new_count_dim_2) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int index_pre=index/count_dim_1;
    int index_dim_1=(index%count_dim_1)/count_after_dim_1;
    int index_insied=(index%count_after_dim_1)/count_dim_2;
    int index_dim_2=index%count_dim_2;
    int new_index=index_pre*new_count_dim_1+
                  index_dim_2*new_count_after_dim_1+
                  index_insied*new_count_dim_2+
                  index_dim_1;
    output_data[new_index]=input_data[index];
  }
}

template <typename Dtype>
void TransposeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  if(bottom[0]->shape().size()>dim_2+1) {
    TransposeDimensionWithLastDimGPU<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
                top[0]->count(), bottom[0]->gpu_data(), top[0]->mutable_gpu_data(), bottom[0]->count(dim_1), 
                bottom[0]->count(dim_1+1), bottom[0]->count(dim_2), bottom[0]->count(dim_2+1), top[0]->count(dim_1), 
                top[0]->count(dim_1+1), top[0]->count(dim_2));
    CUDA_POST_KERNEL_CHECK;
  }
  else {
    TransposeDimensionWithoutLastDimGPU<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
                top[0]->count(), bottom[0]->gpu_data(), top[0]->mutable_gpu_data(), bottom[0]->count(dim_1), 
                bottom[0]->count(dim_1+1), bottom[0]->count(dim_2), top[0]->count(dim_1), 
                top[0]->count(dim_1+1), top[0]->count(dim_2));
    CUDA_POST_KERNEL_CHECK;
  }
}

template <typename Dtype>
void TransposeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if(bottom[0]->shape().size()>dim_2+1) {
    TransposeDimensionWithLastDimGPU<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
                bottom[0]->count(), top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff(), top[0]->count(dim_1), 
                top[0]->count(dim_1+1), top[0]->count(dim_2), top[0]->count(dim_2+1), bottom[0]->count(dim_1), 
                bottom[0]->count(dim_1+1), bottom[0]->count(dim_2));
    CUDA_POST_KERNEL_CHECK;
  }
  else {
    TransposeDimensionWithoutLastDimGPU<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
                bottom[0]->count(), top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff(), top[0]->count(dim_1), 
                top[0]->count(dim_1+1), top[0]->count(dim_2), bottom[0]->count(dim_1), 
                bottom[0]->count(dim_1+1), bottom[0]->count(dim_2));
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(TransposeLayer);
}  // namespace caffe
