#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/mpitask.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + bottom[i]->offset(n), weight,
          top_data + top[i]->offset(n));
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + top[i]->offset(n), bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionDataLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
  }
  Dtype* bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + top[i]->offset(n));
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + bottom[i]->offset(n),
              top_diff + top[i]->offset(n), weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + top[i]->offset(n), weight,
              bottom_diff + bottom[i]->offset(n));
        }
      }
    }
  }
#ifdef STRATEGY_0
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    cudaDeviceSynchronize();
    MPI_Allreduce(MPI_IN_PLACE, bias_diff, this->blobs_[1]->count(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    cudaDeviceSynchronize();
  }
  if (this->param_propagate_down_[0]) {
    cudaDeviceSynchronize();
    MPI_Allreduce(MPI_IN_PLACE, weight_diff, this->blobs_[0]->count(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    cudaDeviceSynchronize();
  }
#else
  MpiTaskList<Dtype> *task_list = (MpiTaskList<Dtype> *)Caffe::getTaskList();
  cudaDeviceSynchronize();
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    task_list->push_back(new MpiTask<Dtype>(NULL,0,this->blobs_[1].get(),1,this->blobs_[1]->count()));
  }
  if (this->param_propagate_down_[0]) {
    task_list->push_back(new MpiTask<Dtype>(NULL,0,this->blobs_[0].get(),1,this->blobs_[0]->count()));
  }
#endif
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionDataLayer);

}  // namespace caffe
