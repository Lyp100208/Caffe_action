#include <cfloat>
#include <vector>
#include <math_functions.h>

#include "thrust/device_vector.h"
#include "caffe/common.hpp"
#include "caffe/sequence_layers.hpp"

#ifdef USE_MPI
#include "caffe/mpitask.hpp"
#endif

namespace caffe {
  template <typename Dtype>
    __device__ Dtype sigmoid(const Dtype x) {
      return Dtype(1) / (Dtype(1) + exp(-x));
    }

  template <typename Dtype>
    __device__ Dtype tanh(const Dtype x) {
      return Dtype(2) * sigmoid(Dtype(2) * x) - Dtype(1);
    }

  template <typename Dtype>
    __global__ void lstm_copy_indicator(const int count, const int output_feature_dim_,
        const Dtype * cont_t, const Dtype * src, Dtype * dst) {
      CUDA_KERNEL_LOOP(i, count) {
        const int b = i / output_feature_dim_;
        dst[i] = (cont_t[b] > 0) ? src[i] : Dtype(0);
      }
    }

  template <typename Dtype>
    __global__ void lstm_forward_kernel(const int count, const int output_feature_dim_,
        Dtype * gates, Dtype * h, Dtype * c,
        const Dtype * c_prev) {
      CUDA_KERNEL_LOOP(index, count) {
        const int index_batch = index / output_feature_dim_,
                  index_feature = index % output_feature_dim_;
        const int offset = index_batch * SLLSTMLayer<Dtype>::NumOfGates
				                  * output_feature_dim_ + index_feature;
        const int fi = SLLSTMLayer<Dtype>::I * output_feature_dim_ + offset,
                  ff = SLLSTMLayer<Dtype>::F * output_feature_dim_ + offset,
                  fo = SLLSTMLayer<Dtype>::O * output_feature_dim_ + offset,
                  fg = SLLSTMLayer<Dtype>::G * output_feature_dim_ + offset;
        gates[fi] = sigmoid(gates[fi]);
        gates[ff] = sigmoid(gates[ff]);
        gates[fo] = sigmoid(gates[fo]);
        gates[fg] = tanh(gates[fg]);

        c[index] = gates[fi] * gates[fg] + gates[ff] * c_prev[index];
        h[index] = gates[fo] * tanh(c[index]);
      }
    }

  template <typename Dtype>
    __global__ void lstm_backward_kernel(const int batch, const int output_feature_dim_,
              const Dtype * gates, Dtype * gates_diff,
              const Dtype * c, const Dtype * c_diff,
              const Dtype * c_prev, Dtype * c_backpropagate,
              const Dtype * h_diff) {
      CUDA_KERNEL_LOOP(index, batch * output_feature_dim_) {
        const int index_batch = index / output_feature_dim_,
                  index_feature = index % output_feature_dim_;
        const int offset = index_batch * SLLSTMLayer<Dtype>::NumOfGates
                          * output_feature_dim_ + index_feature;
        const int fi = SLLSTMLayer<Dtype>::I * output_feature_dim_ + offset,
                  ff = SLLSTMLayer<Dtype>::F * output_feature_dim_ + offset,
                  fo = SLLSTMLayer<Dtype>::O * output_feature_dim_ + offset,
                  fg = SLLSTMLayer<Dtype>::G * output_feature_dim_ + offset;
        const Dtype tanhc = tanh(c[index]);

        gates_diff[fo] = tanhc * h_diff[index];
        Dtype c_term_diff = c_diff[index] + (1 - tanhc * tanhc)
                            * gates[fo] * h_diff[index];
        gates_diff[ff] = c_prev[index] * c_term_diff;
        c_backpropagate[index] = gates[ff] * c_term_diff;
        gates_diff[fi] = gates[fg] * c_term_diff;
        gates_diff[fg] = gates[fi] * c_term_diff;
      }
    }

  template <typename Dtype>
    __global__ void lstm_acts_backward(const int count, const int output_feature_dim_,
              const Dtype * gates, Dtype * gates_diff){
      const int x_dim = SLLSTMLayer<Dtype>::NumOfGates * output_feature_dim_;
      CUDA_KERNEL_LOOP(index, count) {
        const int d = index % x_dim;
        const Dtype x_act = gates[index];
        if (d < 3 * output_feature_dim_)
          gates_diff[index] = x_act * (1 - x_act) * gates_diff[index];
        else
          gates_diff[index] = (1 - x_act * x_act) * gates_diff[index];
      }
    }

  template <typename Dtype>
    void SLLSTMLayer<Dtype>::copy_prev_gpu(int t, int count,
        const Dtype *cont_t,
        const Dtype *c_t, const Dtype *h_t,
        Dtype *c_prev, Dtype *h_prev) {
      if (t > 0) {
        if (cont_t) {
          // NOLINT_NEXT_LINE(whitespace/operators)
          lstm_copy_indicator<Dtype><<<CAFFE_GET_BLOCKS(count),
            CAFFE_CUDA_NUM_THREADS>>>(count, output_feature_dim_,
                cont_t, c_t - count, c_prev);
          CUDA_POST_KERNEL_CHECK;
          // NOLINT_NEXT_LINE(whitespace/operators)
          lstm_copy_indicator<Dtype><<<CAFFE_GET_BLOCKS(count),
            CAFFE_CUDA_NUM_THREADS>>>(count, output_feature_dim_,
                cont_t, h_t - count, h_prev);
          CUDA_POST_KERNEL_CHECK;
        } else {
          caffe_copy(count, c_t - count, c_prev);
          caffe_copy(count, h_t - count, h_prev);
        }
      } else {
        caffe_gpu_set(count, Dtype(0), c_prev);
        caffe_gpu_set(count, Dtype(0), h_prev);
      }
    }

  template <typename Dtype>
    void SLLSTMLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
      const Dtype * x = bottom[0]->gpu_data();
      const Dtype * cont = (bottom.size() > 1) ? bottom[1]->gpu_data() : NULL;
      const Dtype * x_static = (bottom.size() > 2) ? bottom[2]->gpu_data() : NULL;
      const int T = bottom[0]->shape(0);
      const int batch = bottom[0]->shape(1);
      const int count = batch * output_feature_dim_;
      const int hidden_dim_ = NumOfGates * output_feature_dim_;

      const Dtype * wx = this->blobs_[WX]->gpu_data();
      const Dtype * ws = (x_static) ? this->blobs_[WS]->gpu_data() : NULL;
      const Dtype * uh = this->blobs_[UH]->gpu_data();
      const Dtype * b = this->blobs_[B]->gpu_data();

      Dtype * c = cell_.mutable_gpu_data();
      Dtype * gates = gates_.mutable_gpu_data();
      Dtype * h = top[0]->mutable_gpu_data();

      Dtype * c_prev = buffer_c_prev_.mutable_gpu_data();
      Dtype * h_prev = buffer_h_prev_.mutable_gpu_data();

      const Dtype * bias_multiplier = bias_multiplier_.gpu_data();
      Dtype * x_static_ws = (x_static) ? x_static_ws_.mutable_gpu_data() : NULL;

      caffe_gpu_gemm(CblasNoTrans, CblasTrans,
          T * batch, hidden_dim_, input_feature_dim_,
          Dtype(1), x, wx,
          Dtype(0), gates);
      caffe_gpu_gemm(CblasNoTrans, CblasNoTrans,
          T * batch, hidden_dim_, 1,
          Dtype(1), bias_multiplier, b,
          Dtype(1), gates);
      if (x_static)  caffe_gpu_gemm(CblasNoTrans, CblasTrans,
          batch, hidden_dim_, input_feature_dim_,
          Dtype(1), x_static, ws,
          Dtype(0), x_static_ws);

      for (int t = 0; t < T; t++) {
        const Dtype * cont_t = (cont ? cont + t * batch : NULL);
        Dtype * gates_t = gates + t * count * NumOfGates;
        Dtype * c_t = c + t * count;
        Dtype * h_t = h + t * count;

        if (x_static)
          caffe_gpu_add(x_static_ws_.count(), x_static_ws, gates_t, gates_t);

        copy_prev_gpu(t, count, cont_t, c_t, h_t, c_prev, h_prev);

        caffe_gpu_gemm(CblasNoTrans, CblasTrans,
            batch, hidden_dim_, output_feature_dim_,
            Dtype(1), h_prev, uh,
            Dtype(1), gates_t);

        // NOLINT_NEXT_LINE(whitespace/operators)
        lstm_forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(count),
          CAFFE_CUDA_NUM_THREADS>>>(count, output_feature_dim_,
              gates_t, h_t, c_t, c_prev);
        CUDA_POST_KERNEL_CHECK;
      }
    }

  template <typename Dtype>
    void SLLSTMLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

      if (propagate_down.size() > 1) {
        CHECK(!propagate_down[1]) << "Cannot back-propagate to continuous indicator.";
      }

      // clean blobs_[n]->gpu_diff()
      for (int i = 0; i < NumOfBlobs; i++) {
        caffe_gpu_set(this->blobs_[i]->count(), Dtype(0),
            this->blobs_[i]->mutable_gpu_diff());
      }

      const Dtype * x_static = NULL, *ws = NULL;
      Dtype * ws_diff = NULL, *x_static_diff = NULL;
      if (bottom.size() > 2) {
        ws = this->blobs_[WS]->gpu_data();
        ws_diff = this->blobs_[WS]->mutable_gpu_diff();
        x_static = bottom[2]->gpu_data();
        if (propagate_down[2]) {
          x_static_diff = bottom[2]->mutable_gpu_diff();
          caffe_gpu_set(bottom[2]->count(), Dtype(0),
            bottom[2]->mutable_gpu_diff());
        }
      }

      const int T = bottom[0]->shape(0);
      const int batch = bottom[0]->shape(1);
      const int count = batch * output_feature_dim_;
      const int hidden_dim_ = NumOfGates * output_feature_dim_;

      // clean c_prev(and diff) & h_prev(and diff)
      Dtype * c_diff[2];
      c_diff[0] = buffer_c_diff_.mutable_gpu_data();
      c_diff[1] = buffer_c_diff_.mutable_gpu_diff();
      caffe_gpu_set(count, Dtype(0), c_diff[0]);
      caffe_gpu_set(count, Dtype(0), c_diff[1]);
      Dtype * h_prev = buffer_h_prev_.mutable_gpu_data();
      Dtype * h_backpropagate = buffer_h_prev_.mutable_gpu_diff();
      caffe_gpu_set(count, Dtype(0), h_backpropagate);
      Dtype * c_prev = buffer_c_prev_.mutable_gpu_data();

      // pointers
      const Dtype * x = bottom[0]->gpu_data();
      const Dtype * cont = (bottom.size() > 1) ? bottom[1]->gpu_data() : NULL;

      const Dtype * h = top[0]->gpu_data();
      const Dtype * c = cell_.gpu_data();
      const Dtype * gates = gates_.gpu_data();
      const Dtype * wx = this->blobs_[WX]->gpu_data();
      const Dtype * uh = this->blobs_[UH]->gpu_data();
      const Dtype * bias_multiplier = bias_multiplier_.gpu_data();

      Dtype * h_diff = top[0]->mutable_gpu_diff();
      Dtype * gates_diff = gates_.mutable_gpu_diff();
      Dtype * wx_diff = this->blobs_[WX]->mutable_gpu_diff();
      Dtype * uh_diff = this->blobs_[UH]->mutable_gpu_diff();
      Dtype * b_diff = this->blobs_[B]->mutable_gpu_diff();
      Dtype * x_diff = propagate_down[0] ? bottom[0]->mutable_gpu_diff() : NULL;

      bool FLAG = true;
      // loop body
      for (int t = T-1; t >= 0; t--) {
        const Dtype * cont_t = cont ? cont + t * batch : NULL;
        int offset = t * count;
        const Dtype * h_t = h + offset;
        const Dtype * c_t = c + offset;
        const Dtype * gates_t = gates + offset * NumOfGates;
        const Dtype * x_t = x + t * batch * input_feature_dim_;

        Dtype * h_t_diff = h_diff + offset;
        FLAG = !FLAG;
        const Dtype * c_t_diff = c_diff[FLAG];
        Dtype * c_backpropagate = c_diff[!FLAG];
        Dtype * gates_t_diff = gates_diff + offset * NumOfGates;

        // accumulate.
        caffe_gpu_add(count, h_backpropagate, h_t_diff, h_t_diff);

        copy_prev_gpu(t, count, cont_t, c_t, h_t, c_prev, h_prev);

        // NOLINT_NEXT_LINE(whitespace/operators)
        lstm_backward_kernel<Dtype><<<CAFFE_GET_BLOCKS(count),
          CAFFE_CUDA_NUM_THREADS>>>(batch, output_feature_dim_,
              gates_t, gates_t_diff,
              c_t, c_t_diff,
              c_prev, c_backpropagate,
              h_t_diff);
        CUDA_POST_KERNEL_CHECK;
        // NOLINT_NEXT_LINE(whitespace/operators)
        lstm_acts_backward<Dtype><<<CAFFE_GET_BLOCKS(count*NumOfGates),
          CAFFE_CUDA_NUM_THREADS>>>(count*NumOfGates, output_feature_dim_,
              gates_t, gates_t_diff);
        CUDA_POST_KERNEL_CHECK;

        caffe_gpu_gemm(CblasTrans, CblasNoTrans,
            hidden_dim_, input_feature_dim_, batch,
            Dtype(1), gates_t_diff, x_t,
            Dtype(1), wx_diff);
        if (x_static) caffe_gpu_gemm(CblasTrans, CblasNoTrans,
            hidden_dim_, input_feature_dim_, batch,
            Dtype(1), gates_t_diff, x_static,
            Dtype(1), ws_diff);
        if (x_static_diff) caffe_gpu_gemm(CblasNoTrans, CblasNoTrans,
            batch, input_feature_dim_, hidden_dim_,
            Dtype(1), gates_t_diff, ws,
            Dtype(1), x_static_diff);

        caffe_gpu_gemm(CblasTrans, CblasNoTrans,
            hidden_dim_, output_feature_dim_, batch,
            Dtype(1), gates_t_diff, h_prev,
            Dtype(1), uh_diff);
        caffe_gpu_gemm(CblasNoTrans, CblasNoTrans,
            batch, output_feature_dim_, hidden_dim_,
            Dtype(1), gates_t_diff, uh,
            Dtype(0), h_backpropagate);

        if (t > 0 && cont_t) {
          // NOLINT_NEXT_LINE(whitespace/operators)
          lstm_copy_indicator<Dtype><<<CAFFE_GET_BLOCKS(count),
            CAFFE_CUDA_NUM_THREADS>>>(count, output_feature_dim_,
                cont_t, h_backpropagate, h_backpropagate);
          CUDA_POST_KERNEL_CHECK;
          // NOLINT_NEXT_LINE(whitespace/operators)
          lstm_copy_indicator<Dtype><<<CAFFE_GET_BLOCKS(count),
            CAFFE_CUDA_NUM_THREADS>>>(count, output_feature_dim_,
                cont_t, c_backpropagate, c_backpropagate);
          CUDA_POST_KERNEL_CHECK;
        }
      }
      if (x_diff) caffe_gpu_gemm(CblasNoTrans, CblasNoTrans,
        batch*T, input_feature_dim_, hidden_dim_,
        Dtype(1), gates_diff, wx,
        Dtype(0), x_diff);

      caffe_gpu_gemv<Dtype>(CblasTrans, T * batch, hidden_dim_, 1,
          gates_diff, bias_multiplier, Dtype(1), b_diff);

#ifdef USE_MPI
      if (Caffe::getStrategy() == 0) {
        cudaDeviceSynchronize();
        for (int i = 0; i < NumOfBlobs; i++) {
          MPI_Allreduce(MPI_IN_PLACE, this->blobs_[i]->mutable_gpu_diff(),
              this->blobs_[i]->count(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        }
      } else if (Caffe::getStrategy() == 1) {
        MpiTaskList<Dtype> *task_list = (MpiTaskList<Dtype> *)Caffe::getTaskList();
        for (int i = 0; i < NumOfBlobs; i++) {
          this->blobs_[i]->mutable_cpu_diff();
        }
        cudaDeviceSynchronize();
        for (int i = 0; i < NumOfBlobs; i++) {
          task_list->push_back(new MpiTask<Dtype>(NULL,0,this->blobs_[i].get(),1,this->blobs_[i]->count()));
        }
      } else {
      }
#endif // USE_MPI
    }

  INSTANTIATE_LAYER_GPU_FUNCS(SLLSTMLayer);
}; // namespace caffe
