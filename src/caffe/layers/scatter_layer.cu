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
void ScatterLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  cudaDeviceSynchronize();
  for (int i = 0; i < bottom.size(); ++i) {
    MPI_Scatter(bottom[i]->gpu_data(),top[i]->count(),
        MPI_FLOAT,top[i]->mutable_gpu_data(),
        top[i]->count(),MPI_FLOAT,0,MPI_COMM_WORLD);
    cudaDeviceSynchronize();
  }
}

template <typename Dtype>
void ScatterLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    cudaDeviceSynchronize();
    MpiTaskList<Dtype> *task_list = (MpiTaskList<Dtype> *)Caffe::getTaskList();
    task_list->wait_all_task();
    for (int i = 0; i < top.size(); ++i) {
      MPI_Gather(top[i]->gpu_diff(),top[i]->count(),MPI_FLOAT,
          bottom[i]->mutable_gpu_diff(),top[i]->count(),
          MPI_FLOAT,0,MPI_COMM_WORLD);
      cudaDeviceSynchronize();
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ScatterLayer);
}  // namespace caffe
