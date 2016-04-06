#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/mpitask.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductAllLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  cudaDeviceSynchronize();
  MPI_Gather(bottom[0]->gpu_data(),bottom[0]->count(),MPI_FLOAT,bottom_temp_.mutable_gpu_data(),
      bottom[0]->count(),MPI_FLOAT,0,MPI_COMM_WORLD);
  cudaDeviceSynchronize();
  if (Caffe::getThreadId() == 0) {
    const Dtype* bottom_data = bottom_temp_.gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const Dtype* weight = this->blobs_[0]->gpu_data();
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
        bottom_data, weight, (Dtype)0., top_data);
    if (bias_term_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
          bias_multiplier_.gpu_data(),
          this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
    }
  }
}

template <typename Dtype>
void InnerProductAllLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (Caffe::getThreadId() == 0) {
    if (this->param_propagate_down_[0]) {
      const Dtype* top_diff = top[0]->gpu_diff();
      const Dtype* bottom_data = bottom_temp_.gpu_data();
      // Gradient with respect to weight
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
          top_diff, bottom_data, (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    }
    if (bias_term_ && this->param_propagate_down_[1]) {
      const Dtype* top_diff = top[0]->gpu_diff();
      // Gradient with respect to bias
      caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
          bias_multiplier_.gpu_data(), (Dtype)1.,
          this->blobs_[1]->mutable_gpu_diff());
    }
    if (propagate_down[0]) {
      const Dtype* top_diff = top[0]->gpu_diff();
      // Gradient with respect to bottom data
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
          top_diff, this->blobs_[0]->gpu_data(), (Dtype)0.,
          bottom_temp_.mutable_gpu_diff());
    }
  }
  cudaDeviceSynchronize();
  MpiTaskList<Dtype> *task_list = (MpiTaskList<Dtype> *)Caffe::getTaskList();
  task_list->wait_all_task();
  MPI_Scatter(bottom_temp_.gpu_diff(),bottom[0]->count(),MPI_FLOAT,bottom[0]->mutable_gpu_diff(),bottom[0]->count(),
      MPI_FLOAT,0,MPI_COMM_WORLD);
  cudaDeviceSynchronize();
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductAllLayer);

}  // namespace caffe
