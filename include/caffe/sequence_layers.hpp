#ifndef CAFFE_SEQUENCE_LAYERS_HPP_
#define CAFFE_SEQUENCE_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>
#include <tr1/unordered_map>
#include <queue>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * Single-layered LSTM.
 *  1. Bottom blobs:
 *      [0]: required. input sequence.
 *        size: Sequence_length x Batch_size x Feature_dim
 *      [1]: optional. continuous indicator.
 *        size: Sequence_length x Batch_size
 *      [2]: optional. static input.
 *        size: Batch_size x Feature_dim
 *  2. Output blobs:
 *      [0]: output sequence.
 */
template <typename Dtype>
class SLLSTMLayer : public Layer<Dtype> {
public:
  explicit SLLSTMLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SLLSTM"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  virtual inline bool AllowForceBackward(int bottom_index) const {
    // Cannot propagate back to continuous indicator.
    return bottom_index != 1;
  }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

public:
  enum BlobName {
    UH = 0, B, WX, WS
  };

  enum {
    I = 0, F, O, G,
    NumOfGates
  };

protected:
  void copy_prev_gpu(int t, int count, const Dtype *cont_t,
                     const Dtype *c_t, const Dtype *h_t,
                     Dtype *c_prev, Dtype *h_prev);
  void copy_prev_cpu(int t, int count, const Dtype *cont_t,
                     const Dtype *c_t, const Dtype *h_t,
                     Dtype *c_prev, Dtype *h_prev);
  Blob<Dtype> gates_,   // [I. F, O, G]
       cell_,    // C
       buffer_h_prev_,
       buffer_c_prev_,
       buffer_c_diff_,
       x_static_ws_;

  int input_feature_dim_,
      output_feature_dim_,
      NumOfBlobs;
  Blob<Dtype> bias_multiplier_;
};

///**
// * Single-layered GRNN.
// *  1. Bottom blobs:
// *      [0]: required. input sequence.
// *        size: Sequence_length x Batch_size x Feature_dim
// *      [1]: optional. continuous indicator.
// *        size: Sequence_length x Batch_size
// *      [2]: optional. static input.
// *        size: Batch_size x Feature_dim
// *  2. Output blobs:
// *      [0]: output sequence.
// */
//template <typename Dtype>
//class SLGRNNLayer : public Layer<Dtype> {
//public:
//  explicit SLGRNNLayer(const LayerParameter& param)
//    : Layer<Dtype>(param) {}
//  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
//                          const vector<Blob<Dtype>*>& top);
//  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
//                       const vector<Blob<Dtype>*>& top);
//
//  virtual inline const char* type() const { return "SLGRNN"; }
//  virtual inline int MinBottomBlobs() const { return 1; }
//  virtual inline int MaxBottomBlobs() const { return 3; }
//  virtual inline int ExactNumTopBlobs() const { return 1; }
//
//  virtual inline bool AllowForceBackward(int bottom_index) const {
//    // Cannot propagate back to continuous indicator.
//    return bottom_index != 1;
//  }
//
//protected:
//  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
//                           const vector<Blob<Dtype>*>& top);
//  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//                           const vector<Blob<Dtype>*>& top);
//
//  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
//                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
//  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
//                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
//
//public:
//  enum BlobName {
//    UH = 0, B, WX, WS
//  };
//
//  enum {
//    Z = 0, R, G,
//    NumOfGates
//  };
//
//protected:
//  void copy_prev_gpu(int t, int count,
//                     const Dtype *cont_t, const Dtype *h_t, Dtype *h_prev);
//  void copy_prev_cpu(int t, int count,
//                     const Dtype *cont_t, const Dtype *h_t, Dtype *h_prev);
//  Blob<Dtype> gates_,   // [Z. R, G]
//       buffer_h_prev_,
//       x_static_ws_,
//       buffer_uh_;
//
//  int input_feature_dim_,
//      output_feature_dim_,
//      NumOfBlobs;
//  Blob<Dtype> bias_multiplier_;
//};
//
//template <typename Dtype> class RecurrentLayer;
//
///**
// * @brief An abstract class for implementing recurrent behavior inside of an
// *        unrolled network.  This Layer type cannot be instantiated -- instaed,
// *        you should use one of its implementations which defines the recurrent
// *        architecture, such as RNNLayer or LSTMLayer.
// */
//template <typename Dtype>
//class RecurrentLayer : public Layer<Dtype> {
//public:
//  explicit RecurrentLayer(const LayerParameter& param)
//    : Layer<Dtype>(param) {}
//  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
//                          const vector<Blob<Dtype>*>& top);
//  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
//                       const vector<Blob<Dtype>*>& top);
//  virtual void Reset();
//
//  virtual inline const char* type() const { return "Recurrent"; }
//  virtual inline int MinBottomBlobs() const { return 2; }
//  virtual inline int MaxBottomBlobs() const { return 3; }
//  virtual inline int ExactNumTopBlobs() const { return 1; }
//
//  virtual inline bool AllowForceBackward(const int bottom_index) const {
//    // Can't propagate to sequence continuation indicators.
//    return bottom_index != 1;
//  }
//
//protected:
//  /**
//   * @brief Fills net_param with the recurrent network arcthiecture.  Subclasses
//   *        should define this -- see RNNLayer and LSTMLayer for examples.
//   */
//  virtual void FillUnrolledNet(NetParameter* net_param) const = 0;
//
//  /**
//   * @brief Fills names with the names of the 0th timestep recurrent input
//   *        Blob&s.  Subclasses should define this -- see RNNLayer and LSTMLayer
//   *        for examples.
//   */
//  virtual void RecurrentInputBlobNames(vector<string>* names) const = 0;
//
//  /**
//   * @brief Fills shapes with the shapes of the recurrent input Blob&s.
//   *        Subclasses should define this -- see RNNLayer and LSTMLayer
//   *        for examples.
//   */
//  virtual void RecurrentInputShapes(vector<BlobShape>* shapes) const = 0;
//
//  /**
//   * @brief Fills names with the names of the Tth timestep recurrent output
//   *        Blob&s.  Subclasses should define this -- see RNNLayer and LSTMLayer
//   *        for examples.
//   */
//  virtual void RecurrentOutputBlobNames(vector<string>* names) const = 0;
//
//  /**
//   * @brief Fills names with the names of the output blobs, concatenated across
//   *        all timesteps.  Should return a name for each top Blob.
//   *        Subclasses should define this -- see RNNLayer and LSTMLayer for
//   *        examples.
//   */
//  virtual void OutputBlobNames(vector<string>* names) const = 0;
//
//  /**
//   * @param bottom input Blob vector (length 2-3)
//   *
//   *   -# @f$ (T \times N \times ...) @f$
//   *      the time-varying input @f$ x @f$.  After the first two axes, whose
//   *      dimensions must correspond to the number of timesteps @f$ T @f$ and
//   *      the number of independent streams @f$ N @f$, respectively, its
//   *      dimensions may be arbitrary.  Note that the ordering of dimensions --
//   *      @f$ (T \times N \times ...) @f$, rather than
//   *      @f$ (N \times T \times ...) @f$ -- means that the @f$ N @f$
//   *      independent input streams must be "interleaved".
//   *
//   *   -# @f$ (T \times N) @f$
//   *      the sequence continuation indicators @f$ \delta @f$.
//   *      These inputs should be binary (0 or 1) indicators, where
//   *      @f$ \delta_{t,n} = 0 @f$ means that timestep @f$ t @f$ of stream
//   *      @f$ n @f$ is the beginning of a new sequence, and hence the previous
//   *      hidden state @f$ h_{t-1} @f$ is multiplied by @f$ \delta_t = 0 @f$
//   *      and has no effect on the cell's output at timestep @f$ t @f$, and
//   *      a value of @f$ \delta_{t,n} = 1 @f$ means that timestep @f$ t @f$ of
//   *      stream @f$ n @f$ is a continuation from the previous timestep
//   *      @f$ t-1 @f$, and the previous hidden state @f$ h_{t-1} @f$ affects the
//   *      updated hidden state and output.
//   *
//   *   -# @f$ (N \times ...) @f$ (optional)
//   *      the static (non-time-varying) input @f$ x_{static} @f$.
//   *      After the first axis, whose dimension must be the number of
//   *      independent streams, its dimensions may be arbitrary.
//   *      This is mathematically equivalent to using a time-varying input of
//   *      @f$ x'_t = [x_t; x_{static}] @f$ -- i.e., tiling the static input
//   *      across the @f$ T @f$ timesteps and concatenating with the time-varying
//   *      input.  Note that if this input is used, all timesteps in a single
//   *      batch within a particular one of the @f$ N @f$ streams must share the
//   *      same static input, even if the sequence continuation indicators
//   *      suggest that difference sequences are ending and beginning within a
//   *      single batch.  This may require padding and/or truncation for uniform
//   *      length.
//   *
//   * @param top output Blob vector (length 1)
//   *   -# @f$ (T \times N \times D) @f$
//   *      the time-varying output @f$ y @f$, where @f$ D @f$ is
//   *      <code>recurrent_param.num_output()</code>.
//   *      Refer to documentation for particular RecurrentLayer implementations
//   *      (such as RNNLayer and LSTMLayer) for the definition of @f$ y @f$.
//   */
//  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
//                           const vector<Blob<Dtype>*>& top);
//  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//                           const vector<Blob<Dtype>*>& top);
//  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
//                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
//
//  /// @brief A helper function, useful for stringifying timestep indices.
//  virtual string int_to_str(const int t) const;
//
//  /// @brief A Net to implement the Recurrent functionality.
//  shared_ptr<Net<Dtype> > unrolled_net_;
//
//  /// @brief The number of independent streams to process simultaneously.
//  int N_;
//
//  /**
//   * @brief The number of timesteps in the layer's input, and the number of
//   *        timesteps over which to backpropagate through time.
//   */
//  int T_;
//
//  /// @brief Whether the layer has a "static" input copied across all timesteps.
//  bool static_input_;
//
//  vector<Blob<Dtype>* > recur_input_blobs_;
//  vector<Blob<Dtype>* > recur_output_blobs_;
//  vector<Blob<Dtype>* > output_blobs_;
//  Blob<Dtype>* x_input_blob_;
//  Blob<Dtype>* x_static_input_blob_;
//  Blob<Dtype>* cont_input_blob_;
//};
//
///**
// * @brief Processes sequential inputs using a "Long Short-Term Memory" (LSTM)
// *        [1] style recurrent neural network (RNN). Implemented as a network
// *        unrolled the LSTM computation in time.
// *
// *
// * The specific architecture used in this implementation is as described in
// * "Learning to Execute" [2], reproduced below:
// *     i_t := \sigmoid[ W_{hi} * h_{t-1} + W_{xi} * x_t + b_i ]
// *     f_t := \sigmoid[ W_{hf} * h_{t-1} + W_{xf} * x_t + b_f ]
// *     o_t := \sigmoid[ W_{ho} * h_{t-1} + W_{xo} * x_t + b_o ]
// *     g_t :=    \tanh[ W_{hg} * h_{t-1} + W_{xg} * x_t + b_g ]
// *     c_t := (f_t .* c_{t-1}) + (i_t .* g_t)
// *     h_t := o_t .* \tanh[c_t]
// * In the implementation, the i, f, o, and g computations are performed as a
// * single inner product.
// *
// * Notably, this implementation lacks the "diagonal" gates, as used in the
// * LSTM architectures described by Alex Graves [3] and others.
// *
// * [1] Hochreiter, Sepp, and Schmidhuber, Jürgen. "Long short-term memory."
// *     Neural Computation 9, no. 8 (1997): 1735-1780.
// *
// * [2] Zaremba, Wojciech, and Sutskever, Ilya. "Learning to execute."
// *     arXiv preprint arXiv:1410.4615 (2014).
// *
// * [3] Graves, Alex. "Generating sequences with recurrent neural networks."
// *     arXiv preprint arXiv:1308.0850 (2013).
// */
//template <typename Dtype>
//class LSTMLayer : public RecurrentLayer<Dtype> {
//public:
//  explicit LSTMLayer(const LayerParameter& param)
//    : RecurrentLayer<Dtype>(param) {}
//
//  virtual inline const char* type() const { return "LSTM"; }
//
//protected:
//  virtual void FillUnrolledNet(NetParameter* net_param) const;
//  virtual void RecurrentInputBlobNames(vector<string>* names) const;
//  virtual void RecurrentOutputBlobNames(vector<string>* names) const;
//  virtual void RecurrentInputShapes(vector<BlobShape>* shapes) const;
//  virtual void OutputBlobNames(vector<string>* names) const;
//};
//
///**
// * @brief A helper for LSTMLayer: computes a single timestep of the
// *        non-linearity of the LSTM, producing the updated cell and hidden
// *        states.
// */
//template <typename Dtype>
//class LSTMUnitLayer : public Layer<Dtype> {
//public:
//  explicit LSTMUnitLayer(const LayerParameter& param)
//    : Layer<Dtype>(param) {}
//  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
//                          const vector<Blob<Dtype>*>& top);
//  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
//                       const vector<Blob<Dtype>*>& top);
//
//  virtual inline const char* type() const { return "LSTMUnit"; }
//  virtual inline int ExactNumBottomBlobs() const { return 4; }
//  virtual inline int ExactNumTopBlobs() const { return 2; }
//
//  virtual inline bool AllowForceBackward(const int bottom_index) const {
//    // Can't propagate to sequence continuation indicators.
//    return bottom_index != 3;
//  }
//
//protected:
//  /**
//   * @param bottom input Blob vector (length 3)
//   *   -# @f$ (1 \times N \times D) @f$
//   *      the previous timestep cell state @f$ c_{t-1} @f$
//   *   -# @f$ (1 \times N \times D) @f$
//   *      the previous timestep hidden state @f$ h_{t-1} @f$
//   *   -# @f$ (1 \times N \times 4D) @f$
//   *      the X_xc_x @f$ X_xc_x @f$
//   *   -# @f$ (1 \times 1 \times N) @f$
//   *      the sequence continuation indicators  @f$ \delta_t @f$
//   * @param top output Blob vector (length 2)
//   *   -# @f$ (1 \times N \times D) @f$
//   *      the updated cell state @f$ c_t @f$, computed as:
//   *          i_t := \sigmoid[i_t']
//   *          f_t := \sigmoid[f_t']
//   *          o_t := \sigmoid[o_t']
//   *          g_t := \tanh[g_t']
//   *          c_t := cont_t * (f_t .* c_{t-1}) + (i_t .* g_t)
//   *   -# @f$ (1 \times N \times D) @f$
//   *      the updated hidden state @f$ h_t @f$, computed as:
//   *          h_t := o_t .* \tanh[c_t]
//   */
//  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
//                           const vector<Blob<Dtype>*>& top);
//  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//                           const vector<Blob<Dtype>*>& top);
//
//  /**
//   * @brief Computes the error gradient w.r.t. the LSTMUnit inputs.
//   *
//   * @param top output Blob vector (length 2), providing the error gradient with
//   *        respect to the outputs
//   *   -# @f$ (1 \times N \times D) @f$:
//   *      containing error gradients @f$ \frac{\partial E}{\partial c_t} @f$
//   *      with respect to the updated cell state @f$ c_t @f$
//   *   -# @f$ (1 \times N \times D) @f$:
//   *      containing error gradients @f$ \frac{\partial E}{\partial h_t} @f$
//   *      with respect to the updated cell state @f$ h_t @f$
//   * @param propagate_down see Layer::Backward.
//   * @param bottom input Blob vector (length 3), into which the error gradients
//   *        with respect to the LSTMUnit inputs @f$ c_{t-1} @f$ and the gate
//   *        inputs are computed.  Computatation of the error gradients w.r.t.
//   *        the sequence indicators is not implemented.
//   *   -# @f$ (1 \times N \times D) @f$
//   *      the error gradient w.r.t. the previous timestep cell state
//   *      @f$ c_{t-1} @f$
//   *   -# @f$ (1 \times N \times 4D) @f$
//   *      the error gradient w.r.t. the "gate inputs"
//   *      @f$ [
//   *          \frac{\partial E}{\partial i_t}
//   *          \frac{\partial E}{\partial f_t}
//   *          \frac{\partial E}{\partial o_t}
//   *          \frac{\partial E}{\partial g_t}
//   *          ] @f$
//   *   -# @f$ (1 \times 1 \times N) @f$
//   *      the gradient w.r.t. the sequence continuation indicators
//   *      @f$ \delta_t @f$ is currently not computed.
//   */
//  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
//                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
//  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
//                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
//
//  /// @brief The hidden and output dimension.
//  int hidden_dim_;
//  Blob<Dtype> X_acts_;
//};
//
///**
// * @brief Processes sequential inputs using a "Gated Recurrent Neural Network"
// *        [1] style recurrent neural network (RNN). Implemented as a network
// *        unrolled the GRNN computation in time.
// *
// * The specific architecture used in this implementation is reproduced below:
// *    z_t := \sigmoid[ U_{hz} * h_{t-1} + W_{xz} * x_t + b_z ]
// *    r_t := \sigmoid[ U_{hr} * h_{t-1} + W_{xr} * x_t + b_r ]
// *    g_t := \tanh[ r_t .* ( U_{hg} * h_{t-1} ) + W_{xg} * x_t ]
// *    h_t := ( z_t .* h_{t-1} ) + ( (1 - z_t) .* g_t )
// *
// * Other references are:
// *  - KyungHyun Cho's original paper on RNN Encoder-Decoder[2], which is later
// *    simplified and renamed Gated RNN.
// *
// * [1] Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, Yoshua Bengio,
// *      "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence
// *      Modeling." arXiv preprint arXiv:1412.3555v1 (2014)
// *
// * [2] KyungHyun Cho, Bart van Merrienboer, Caglar Gulcehre et al., "Learning
// *      Phrase Representation using RNN Encoder-Decoder for Statistical Machine
// *      Translation." arXiv preprint arXiv:1406.1078v3 (2014)
// */
//template <typename Dtype>
//class GRNNLayer : public RecurrentLayer<Dtype> {
//public:
//  explicit GRNNLayer(const LayerParameter& param)
//    : RecurrentLayer<Dtype>(param) {}
//
//  virtual inline const char* type() const { return "GRNN"; }
//
//protected:
//  virtual void FillUnrolledNet(NetParameter* net_param) const;
//  virtual void RecurrentInputBlobNames(vector<string>* names) const;
//  virtual void RecurrentOutputBlobNames(vector<string>* names) const;
//  virtual void RecurrentInputShapes(vector<BlobShape>* shapes) const;
//  virtual void OutputBlobNames(vector<string>* names) const;
//};
//
///*
// * @brief Helper class for GRNNLayer: computes a single timestep of the
// *    non-linearity of the GRNN, producing the updated cell and hidden states.
// */
//template <typename Dtype>
//class GRNNUnitLayer : public Layer<Dtype> {
//public:
//  explicit GRNNUnitLayer(const LayerParameter& param)
//    : Layer<Dtype>(param) {}
//  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
//                          const vector<Blob<Dtype>*>& top);
//  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
//                       const vector<Blob<Dtype>*>& top);
//
//  virtual inline const char* type() const { return "GRNNUnit"; }
//  virtual inline int ExactNumBottomBlobs() const { return 3; }
//  virtual inline int ExactNumTopBlobs() const { return 1; }
//
//  virtual inline bool AllowForceBackward(const int bottom_index) const {
//    // Can't propagate to sequence continuation indicators.
//    return bottom_index != 2;
//  }
//protected:
//  /**
//   *  @param bottom input Blob vector (length 3)
//   *    -# @f$ (1 \times N \times D) @f$
//   *        the previous timestep hidden state @f$ h_{t-1} @f$
//   *    -# @f$ (1 \times N \times 3D) @f$
//   *        the X_xc_x @f$ X_xc_x @f$
//   *    -# @f$ (1 \times 1 \times N) @f$
//   *        the sequence continuation indicators @f$ \delta_t @f$
//   *  @param top output Blob vector (length 1)
//   *    -# @f$ (1 \times N \times D) @f$
//   *        the updated hidden state
//   */
//  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
//                           const vector<Blob<Dtype>*>& top);
//  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//                           const vector<Blob<Dtype>*>& top);
//
//  /**
//   *  @brief Computes the error gradient w.r.t. the GRNNUnit inputs.
//   *
//   *  @param top output Blob vector (length 1), providing the error gradient
//   *      w.r.t. the outputs.
//   *    -# @f$ (1 \times N \times D) @f$
//   *        the error gradient @f$ \frac{\partial E}{\partial h_t} w.r.t.
//   *        the updated hidden state @f$ h_t @f$
//   *  @param propagate_down see Layer::Backward.
//   *  @param bottom input Blob vector (length 3), into which the error
//   *        gradients w.r.t. the GRNNUnit inputs @f$ h_{t-1} @f$ and the gate
//   *        inputs are computed. Computation of the error gradients w.r.t.
//   *        the sequence indicators is not implemented.
//   *    -# @f$ (1 \times N \times D) @f$
//   *        the error gradient w.r.t. the previous timestep hidden state
//   *        @f$ h_{t-1} @f$
//   *    -# @f$ (1 \times N \times 3D) @f$
//   *        the error gradient w.r.t. the "gate inputs"
//   *        @f$ [
//   *            \frac{\partial E}{\partial r_t}
//   *            \frac{\partial E}{\partial z_t}
//   *            \frac{\partial E}{\partial g_t}
//   *            ] @f$
//   *    -# @f$ (1 \times 1 \times N) @f$
//   *        the gradient w.r.t. the sequence continuation indicators
//   *        @f$ \delta_t @f$ is currently not computed.
//   */
//  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
//                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
//  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
//                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
//
//  /// @brief The hidden and output dimension
//  int num_instances_;
//  int hidden_dim_;
//  int x_dim_;             // always equal to 3*hidden_dim_!
//  Blob<Dtype> h_cont_;    // for buffering h_cont (h_prev .* cont)
//  Blob<Dtype> buffer_;    // buffer for U*h_cont(data) and delta_{z, r, g}(diff)
//};
//
///**
// * @brief Processes time-varying inputs using a simple recurrent neural network
// *        (RNN). Implemented as a network unrolling the RNN computation in time.
// *
// * Given time-varying inputs @f$ x_t @f$, computes hidden state @f$
// *     h_t := \tanh[ W_{hh} h_{t_1} + W_{xh} x_t + b_h ]
// * @f$, and outputs @f$
// *     o_t := \tanh[ W_{ho} h_t + b_o ]
// * @f$.
// */
//template <typename Dtype>
//class RNNLayer : public RecurrentLayer<Dtype> {
//public:
//  explicit RNNLayer(const LayerParameter& param)
//    : RecurrentLayer<Dtype>(param) {}
//
//  virtual inline const char* type() const { return "RNN"; }
//
//protected:
//  virtual void FillUnrolledNet(NetParameter* net_param) const;
//  virtual void RecurrentInputBlobNames(vector<string>* names) const;
//  virtual void RecurrentOutputBlobNames(vector<string>* names) const;
//  virtual void RecurrentInputShapes(vector<BlobShape>* shapes) const;
//  virtual void OutputBlobNames(vector<string>* names) const;
//};
//
//
//#define EPS Dtype(1e-20)
//#define BLANK 0
//
///**
// * @brief Implement CTC (Connectionist Temporal Classification) loss function in [1].
// *
// * [1] Graves A, Fernández S, Gomez F, et al. Connectionist temporal
// *     classification: labelling unsegmented sequence data with recurrent
// *     neural networks[C]//Proceedings of the 23rd international
// *     conference on Machine learning. ACM, 2006: 369-376.
// */
//template <typename Dtype>
//class StandardCTCLayer : public Layer<Dtype> {
//public:
//  explicit StandardCTCLayer(const LayerParameter& param)
//    : Layer<Dtype>(param) {}
//  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
//                          const vector<Blob<Dtype>*>& top);
//  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
//                       const vector<Blob<Dtype>*>& top);
//
//  virtual inline const char* type() const { return "StandardCTC"; }
//  virtual inline int MaxBottomBlobs() const { return 3; }
//  virtual inline int MinBottomBlobs() const { return 2; }
//  virtual inline int MaxTopBlobs() const { return 3; }
//  virtual inline int MinTopBlobs() const { return 1; }
//protected:
//  /// @copydoc StandardCTCLayer
//  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
//                           const vector<Blob<Dtype>*>& top);
//  typedef std::tr1::unordered_map<int, vector<int> > tmap;
//  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
//                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
//  void Backward_internal(const vector<Blob<Dtype>*>& top,
//                         const vector<Blob<Dtype>*>& bottom);
//  void BestPathDecode(Dtype *target_seq, const Dtype *y, const int tt,
//                      Dtype *target_indicator, Dtype *target_score);
//  void BestPathThresDecode(Dtype *target_seq, const Dtype *y, const int tt,
//                           Dtype *target_indicator, Dtype *target_score);
//  void PrefixSearchDecode(Dtype *target_seq, const Dtype *y, const int tt,
//                          Dtype *target_indicator, Dtype *target_score);
//  void PrefixSearchDecode_inner(Dtype * &target_seq, const Dtype *y, const int tt,
//                                Dtype *&target_indicator, Dtype *&target_score);
//  int T, N, C, gap_per_T, thread_num;
//  float threshold;
//  CTCParameter_Decoder decode_type;
//};
///**
// * @brief Implement RNN transduction loss function in [1].
// *
// * [1] Graves A. Sequence transduction with recurrent neural networks[J].
// *     arXiv preprint arXiv:1211.3711, 2012.
// */
//template <typename Dtype>
//class TransductionLayer : public Layer<Dtype> {
//public:
//  explicit TransductionLayer(const LayerParameter& param)
//    : Layer<Dtype>(param) {}
//  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
//                          const vector<Blob<Dtype>*>& top);
//  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
//                       const vector<Blob<Dtype>*>& top);
//
//  virtual inline const char* type() const { return "StandardCTC"; }
//  virtual inline int MaxBottomBlobs() const { return 3; }
//  virtual inline int MinBottomBlobs() const { return 2; }
//  virtual inline int MaxTopBlobs() const { return 1; }
//  virtual inline int MinTopBlobs() const { return 1; }
//protected:
//  /// @copydoc TransductionLayer
//  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
//                           const vector<Blob<Dtype>*>& top);
//  typedef std::tr1::unordered_map<int, vector<int> > tmap;
//  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
//                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
//  void Backward_internal(const vector<Blob<Dtype>*>& top,
//                         const vector<Blob<Dtype>*>& bottom);
//  int T, U, N, C, gap_per_T, thread_num;
//  float threshold;
//};
//
///**
// * @brief Implement RNN transducer loss function & decoder in [1].
// *
// *  IMPORTANT: The proposed transducer contains three sub-networks: transcrip-
// *              tion network, prediction network, and the decoding network. A-
// *              mong which, the transcription network is expected to be defin-
// *              ed outside this layer, which should be a bi-directional RNN l-
// *              ayer (better LSTM).
// *
// * [1] Graves A. Sequence transduction with recurrent neural networks[J].
// *     arXiv preprint arXiv:1211.3711, 2012.
// * [2] Graves A. Speech Recognition with Deep Recurrent Neural Networks.
// *
// *  1. bottom:
// *    [0]: [required] features from transcription network(output of bi-direct-
// *            ional LSTM)
// *        size: sequence_length x batch_size x feature_dim
// *    [1]: [required] continuous indicator. A negative number indicates the s-
// *            tart point of a sequence, the absolute value of this number ind-
// *            icates the length of the sequence. Positive 1 indicates continu-
// *            ous sequence. 0 indicates not a sequence since the point.
// *        size: sequence_length x batch_size
// *    [2]: [required when training] groundtruth. A positive number indicates a
// *            valid label. A 0 means null, which should never appear in groun-
// *            dtruth. A negative 1 indicates not a sequence since the point.
// *        size: sequence_length x batch_size
// *
// *  2. top:
// *    [0]: [required] loss when training / predicted sequence when testing.
// *
// *  3. blobs_:
// *    [0]: W[i, f, o, g]: 4*inner_feature_dim x label_count. for LSTM unit.
// *    [1]: U[i, f, o, g]: 4*inner_feature_dim x (label_count+1). for LSTM unit.
// *    [2]: B[i, f, o, g]: 4*inner_feature_dim. for LSTM unit.
// *    * [3]: W_lh: inner_prob_dim x input_feature_dim
// *    * [4]: W_ph: inner_prob_dim x inner_feature_dim
// *    * [5]: B_h: inner_prob_dim
// *    * [6]: W_hy: (label_count+1) x inner_prob_dim
// *    * [7]: B_y: label_count+1
// *    note: Items with '*' are currently not used. They are reserved for imple-
// *        menting the algorithm described in [2].
// */
//template <typename Dtype>
//class RNNTransducerLayer : public Layer<Dtype> {
//public:
//  explicit RNNTransducerLayer(const LayerParameter& param)
//    : Layer<Dtype>(param) {}
//  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
//                          const vector<Blob<Dtype>*>& top);
//  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
//                       const vector<Blob<Dtype>*>& top);
//
//  virtual inline const char* type() const { return "RNNTransducer"; }
//  virtual inline int MaxBottomBlobs() const { return 3; }
//  virtual inline int MinBottomBlobs() const { return 2; }
//  virtual inline int ExactNumTopBlobs() const { return 1; }
//
//protected:
//  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
//                           const vector<Blob<Dtype>*>& top);
//  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
//                            const vector<bool>& propagate_down,
//                            const vector<Blob<Dtype>*>& bottom);
//
//protected:
//  class DecodeNode {
//  public:
//    Dtype prob;
//    vector<int> target_seq;
//    Blob<Dtype> lstm_buffer;  // [c, h], 2*inner_feature_dim
//
//  public:
//    bool operator<(const DecodeNode& _node) {
//      return this->prob < _node.prob;
//    }
//    bool IsPrefixOf(const DecodeNode& _node) {
//      if (this->target_seq.size() > _node.target_seq.size()) return false;
//      for (vector<int>::const_iterator mit = this->target_seq.begin(),
//           oit = _node.target_seq.begin();
//           mit != this->target_seq.end(); ++mit, ++oit) {
//        if (*mit != *oit) return false;
//      }
//      return true;
//    }
//  };
//
//  vector<DecodeNode> beam_search_, beam_search_tmp_;
//
//protected:
//  vector<int> encode_length_, // actual length of the input sequence
//         // batch_size
//         decode_length_; // actual length of the groundtruth / decoded sequence
//  // batch_size
//  Blob<Dtype> lstm_buffer_,   // [i, o, f, g, c, h],
//       // sequence_length x batch_size x 6*inner_feature_dim
//       // h_,             // h(t,u)
//       //                 // sequence_length x sequence_length x batch_size x inner_prob_dim
//       prob_,          // Pr(k, t, u)
//       // sequence_length x sequence_length x batch_size x (label_count+1)
//       alpha_,         // \alpha(t,u)
//       // sequence_length x sequence_length
//       beta_;          // \beta(t,u)
//  // sequence_length x sequence_length
//  int input_feature_dim_,
//      inner_feature_dim_,
//      inner_prob_dim_,
//      batch_size_,
//      sequence_length_,
//      label_count_,
//      beam_width_;
//};
}  // namespace caffe

#endif  // CAFFE_SEQUENCE_LAYERS_HPP_
