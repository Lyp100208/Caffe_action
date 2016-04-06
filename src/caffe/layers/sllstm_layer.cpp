#include <cmath>

#include "caffe/sequence_layers.hpp"
#include "caffe/filler.hpp"

#ifdef USE_MPI
#include "caffe/mpitask.hpp"
#endif

namespace caffe {
  template <typename Dtype>
    inline Dtype sigmoid(Dtype x) {
      return Dtype(1) / (Dtype(1) + exp(-x));
    }

  template <typename Dtype>
    inline Dtype tanh(Dtype x) {
      return Dtype(2) * sigmoid(Dtype(2) * x) - Dtype(1);
    }

  template <typename Dtype>
    void SLLSTMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
      // feature dim
      input_feature_dim_ = bottom[0]->shape(2);
      output_feature_dim_ = this->layer_param_.recurrent_param().num_output();

      // blobs
      NumOfBlobs = bottom.size() == 3 ? 4 : 3;
      if (this->blobs_.size() > 0) {
        LOG(INFO) << this->layer_param_.name() << " Skipping parameter initialization.";
      } else {
        this->blobs_.resize(NumOfBlobs);
        // WX, WS
        vector<int> shape(2);
        shape[0] = NumOfGates * output_feature_dim_;
        shape[1] = input_feature_dim_;
        for (int i = WX; i < NumOfBlobs; i++) {
          this->blobs_[i].reset(new Blob<Dtype>(shape));
          if (Caffe::getThreadId() == 0) {
            shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
                  this->layer_param_.recurrent_param().weight_filler()));
            weight_filler->Fill(this->blobs_[i].get());
          }
#ifdef USE_MPI
          MPI_Bcast(this->blobs_[i]->mutable_cpu_data(), this->blobs_[i]->count(),
              MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
        }
        // UH
        shape[1] = output_feature_dim_;
        this->blobs_[UH].reset(new Blob<Dtype>(shape));
        if (Caffe::getThreadId() == 0) {
          shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
                this->layer_param_.recurrent_param().weight_filler()));
          weight_filler->Fill(this->blobs_[UH].get());
        }
#ifdef USE_MPI
        MPI_Bcast(this->blobs_[UH]->mutable_cpu_data(), this->blobs_[UH]->count(),
            MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
        // B
        shape.resize(1);
        shape[0] = NumOfGates * output_feature_dim_;
        this->blobs_[B].reset(new Blob<Dtype>(shape));
        if (Caffe::getThreadId() == 0) {
          shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
                this->layer_param_.recurrent_param().bias_filler()));
          weight_filler->Fill(this->blobs_[B].get());
        }
#ifdef USE_MPI
        MPI_Bcast(this->blobs_[B]->mutable_cpu_data(), this->blobs_[B]->count(),
            MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
        this->param_propagate_down_.resize(this->blobs_.size(), true);
      }
    }

  template <typename Dtype>
    void SLLSTMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
      vector<int> shape(bottom[0]->shape());
      shape[2] = output_feature_dim_;
      top[0]->Reshape(shape);
      cell_.Reshape(shape);
      shape[2] = NumOfGates * output_feature_dim_;
      gates_.Reshape(shape);
      if (bottom.size() == 3) {
        CHECK_EQ(NumOfBlobs, 4);
        shape[0] = 1;
        x_static_ws_.Reshape(shape);
      } else CHECK_EQ(NumOfBlobs, 3);
      shape[0] = shape[1];
      shape[1] = output_feature_dim_;
      shape.resize(2);
      buffer_h_prev_.Reshape(shape);
      buffer_c_prev_.Reshape(shape);
      buffer_c_diff_.Reshape(shape);
      vector<int> bias_shape(1, bottom[0]->num() * bottom[0]->channels());
      bias_multiplier_.Reshape(bias_shape);
      caffe_set(bias_multiplier_.count(), Dtype(1), bias_multiplier_.mutable_cpu_data());
    }

  template <typename Dtype>
    void SLLSTMLayer<Dtype>::copy_prev_cpu(int t, int count,
        const Dtype *cont_t,
        const Dtype *c_t, const Dtype *h_t,
        Dtype *c_prev, Dtype *h_prev) {
        if (t > 0) {
          if (cont_t) {
            int batch = count / output_feature_dim_;
            for (int i = 0; i < batch; i++) {
              if (cont_t[i] > 0) {
                caffe_copy(output_feature_dim_, c_t - count + i * output_feature_dim_,
                    c_prev + i * output_feature_dim_);
                caffe_copy(output_feature_dim_, h_t - count + i * output_feature_dim_,
                    h_prev + i * output_feature_dim_);
              } else {
                caffe_set(output_feature_dim_, Dtype(0), c_prev + i * output_feature_dim_);
                caffe_set(output_feature_dim_, Dtype(0), h_prev + i * output_feature_dim_);
              }
            }
          } else {
            caffe_copy(count, c_t - count, c_prev);
            caffe_copy(count, h_t - count, h_prev);
          }
        } else {
          caffe_set(count, Dtype(0), c_prev);
          caffe_set(count, Dtype(0), h_prev);
        }
      }
  template <typename Dtype>
    void SLLSTMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
      const Dtype * x = bottom[0]->cpu_data();
      const Dtype * cont = (bottom.size() > 1) ? bottom[1]->cpu_data() : NULL;
      const Dtype * x_static = (bottom.size() > 2) ? bottom[2]->cpu_data() : NULL;
      const int T = bottom[0]->shape(0);
      const int batch = bottom[0]->shape(1);
      const int count = batch * output_feature_dim_;
      const int hidden_dim_ = NumOfGates * output_feature_dim_;

      const Dtype * wx = this->blobs_[WX]->cpu_data();
      const Dtype * ws = (x_static) ? this->blobs_[WS]->cpu_data() : NULL;
      const Dtype * uh = this->blobs_[UH]->cpu_data();
      const Dtype * b = this->blobs_[B]->cpu_data();

      Dtype * c = cell_.mutable_cpu_data();
      Dtype * gates = gates_.mutable_cpu_data();
      Dtype * h = top[0]->mutable_cpu_data();

      Dtype * c_prev = buffer_c_prev_.mutable_cpu_data();
      Dtype * h_prev = buffer_h_prev_.mutable_cpu_data();

      const Dtype * bias_multiplier = bias_multiplier_.cpu_data();
      Dtype * x_static_ws = (x_static) ? x_static_ws_.mutable_cpu_data() : NULL;

      caffe_cpu_gemm(CblasNoTrans, CblasTrans,
          T * batch, hidden_dim_, input_feature_dim_,
          Dtype(1), x, wx,
          Dtype(0), gates);
      caffe_cpu_gemm(CblasNoTrans, CblasNoTrans,
          T * batch, hidden_dim_, 1,
          Dtype(1), bias_multiplier, b,
          Dtype(1), gates);
      if (x_static)  caffe_cpu_gemm(CblasNoTrans, CblasTrans,
          batch, hidden_dim_, input_feature_dim_,
          Dtype(1), x_static, ws,
          Dtype(0), x_static_ws);

      for (int t = 0; t < T; t++) {
        const Dtype * cont_t = (cont ? cont + t * batch : NULL);
        Dtype * gates_t = gates + t * NumOfGates * count;
        Dtype * c_t = c + t * count;
        Dtype * h_t = h + t * count;

        if (x_static)
          caffe_add(x_static_ws_.count(), x_static_ws, gates_t, gates_t);

        copy_prev_cpu(t, count, cont_t, c_t, h_t, c_prev, h_prev);

        caffe_cpu_gemm(CblasNoTrans, CblasTrans,
            batch, hidden_dim_, output_feature_dim_,
            Dtype(1), h_prev, uh,
            Dtype(1), gates_t);

        for (int i = 0; i < batch; i++) {
          for (int f = 0; f < output_feature_dim_; f++) {
            const int offset = i * hidden_dim_ + f,
                      index = i * output_feature_dim_ + f;
            const int fi = I * output_feature_dim_ + offset,
                      ff = F * output_feature_dim_ + offset,
                      fo = O * output_feature_dim_ + offset,
                      fg = G * output_feature_dim_ + offset;
            gates_t[fi] = sigmoid(gates_t[fi]);
            gates_t[ff] = sigmoid(gates_t[ff]);
            gates_t[fo] = sigmoid(gates_t[fo]);
            gates_t[fg] = tanh(gates_t[fg]);

            c_t[index] = gates_t[fi] * gates_t[fg] + gates_t[ff] * c_prev[index];
            h_t[index] = gates_t[fo] * tanh(c_t[index]);
          }
        }
      }
    }

  template <typename Dtype>
    void SLLSTMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

      if (propagate_down.size() > 1) {
        CHECK(!propagate_down[1]) << "Cannot back-propagate to continuous indicator.";
      }
      if (!propagate_down[0] && (propagate_down.size() < 3 || !propagate_down[2])) {
        return;
      }

      // clean blobs_[n]->cpu_diff()
      for (int i = 0; i < NumOfBlobs; i++) {
        caffe_set(this->blobs_[i]->count(), Dtype(0),
            this->blobs_[i]->mutable_cpu_diff());
      }

      const Dtype * x_static = NULL, *ws = NULL;
      Dtype * ws_diff = NULL, *x_static_diff = NULL;
      if (bottom.size() > 2) {
        ws = this->blobs_[WS]->cpu_data();
        ws_diff = this->blobs_[WS]->mutable_cpu_diff();
        x_static = bottom[2]->cpu_data();
        if (propagate_down[2]) {
          x_static_diff = bottom[2]->mutable_cpu_diff();
          caffe_set(bottom[2]->count(), Dtype(0),
            bottom[2]->mutable_cpu_diff());
        }
      }

      const int T = bottom[0]->shape(0);
      const int batch = bottom[0]->shape(1);
      const int count = batch * output_feature_dim_;

      // clean c_prev(and diff) & h_prev(and diff)
      Dtype * c_diff[2];
      c_diff[0] = buffer_c_diff_.mutable_cpu_data();
      c_diff[1] = buffer_c_diff_.mutable_cpu_diff();
      caffe_set(count, Dtype(0), c_diff[0]);
      caffe_set(count, Dtype(0), c_diff[1]);
      Dtype * h_prev = buffer_h_prev_.mutable_cpu_data();
      Dtype * h_backpropagate = buffer_h_prev_.mutable_cpu_diff();
      caffe_set(count, Dtype(0), h_backpropagate);
      Dtype * c_prev = buffer_c_prev_.mutable_cpu_data();

      // pointers
      const Dtype * x = bottom[0]->cpu_data();
      const Dtype * cont = (bottom.size() > 1) ? bottom[1]->cpu_data() : NULL;

      const Dtype * h = top[0]->cpu_data();
      const Dtype * c = cell_.cpu_data();
      const Dtype * gates = gates_.cpu_data();
      const Dtype * wx = this->blobs_[WX]->cpu_data();
      const Dtype * uh = this->blobs_[UH]->cpu_data();
      const Dtype * bias_multiplier = bias_multiplier_.cpu_data();

      Dtype * h_diff = top[0]->mutable_cpu_diff();
      Dtype * gates_diff = gates_.mutable_cpu_diff();
      Dtype * wx_diff = this->blobs_[WX]->mutable_cpu_diff();
      Dtype * uh_diff = this->blobs_[UH]->mutable_cpu_diff();
      Dtype * b_diff = this->blobs_[B]->mutable_cpu_diff();
      Dtype * x_diff = propagate_down[0] ? bottom[0]->mutable_cpu_diff() : NULL;

      bool FLAG = true;
      const int hidden_dim_ = NumOfGates * output_feature_dim_;
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

        caffe_add(count, h_backpropagate, h_t_diff, h_t_diff);

        copy_prev_cpu(t, count, cont_t, c_t, h_t, c_prev, h_prev);

        for (int i = 0; i < batch; i++) {
          for (int f = 0; f < output_feature_dim_; f++) {
            const int offset = i * hidden_dim_ + f,
                      index = i * output_feature_dim_ + f;
            const int fi = I * output_feature_dim_ + offset,
                      ff = F * output_feature_dim_ + offset,
                      fo = O * output_feature_dim_ + offset,
                      fg = G * output_feature_dim_ + offset;
            const Dtype tanhc = tanh(c_t[index]);

            Dtype c_term_diff = c_t_diff[index] + (1 - tanhc * tanhc)
                              * gates_t[fo] * h_t_diff[index];
            gates_t_diff[fi]  = gates_t[fg] * c_term_diff
                              * gates_t[fi] * (1 - gates_t[fi]);
            gates_t_diff[ff]  = c_prev[index] * c_term_diff
                              * gates_t[ff] * (1 - gates_t[ff]);
            gates_t_diff[fo]  = tanhc * h_t_diff[index]
                              * gates_t[fo] * (1 - gates_t[fo]);
            gates_t_diff[fg]  = gates_t[fi] * c_term_diff
                              * (1 - gates_t[fg] * gates_t[fg]);
            c_backpropagate[index] = gates_t[ff] * c_term_diff;
          }
        }

        caffe_cpu_gemm(CblasTrans, CblasNoTrans,
            hidden_dim_, input_feature_dim_, batch,
            Dtype(1), gates_t_diff, x_t,
            Dtype(1), wx_diff);
        if (x_static) caffe_cpu_gemm(CblasTrans, CblasNoTrans,
            hidden_dim_, input_feature_dim_, batch,
            Dtype(1), gates_t_diff, x_static,
            Dtype(1), ws_diff);
        if (x_static_diff) caffe_cpu_gemm(CblasNoTrans, CblasNoTrans,
            batch, input_feature_dim_, hidden_dim_,
            Dtype(1), gates_t_diff, ws,
            Dtype(1), x_static_diff);
        caffe_cpu_gemm(CblasTrans, CblasNoTrans,
            hidden_dim_, output_feature_dim_, batch,
            Dtype(1), gates_t_diff, h_prev,
            Dtype(1), uh_diff);
        caffe_cpu_gemm(CblasNoTrans, CblasNoTrans,
            batch, output_feature_dim_, hidden_dim_,
            Dtype(1), gates_t_diff, uh,
            Dtype(0), h_backpropagate);

        if (t > 0 && cont_t) {
          for (int i = 0; i < batch; i++) {
            if (cont_t[i] <= 0) {
              caffe_set(output_feature_dim_, Dtype(0), h_backpropagate + i * output_feature_dim_);
              caffe_set(output_feature_dim_, Dtype(0), c_backpropagate + i * output_feature_dim_);
            }
          }
        }
      }
      if (x_diff) caffe_cpu_gemm(CblasNoTrans, CblasNoTrans,
          batch*T, input_feature_dim_, hidden_dim_,
          Dtype(1), gates_diff, wx,
          Dtype(0), x_diff);

      caffe_cpu_gemv<Dtype>(CblasTrans, T * batch, hidden_dim_, 1,
          gates_diff, bias_multiplier, Dtype(1), b_diff);
    }

#ifdef CPU_ONLY
  STUB_GPU(SLLSTMLayer);
#endif

  INSTANTIATE_CLASS(SLLSTMLayer);
  REGISTER_LAYER_CLASS(SLLSTM);
} // namespace caffe
