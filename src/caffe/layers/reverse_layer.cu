#include "caffe/common_layers.hpp"

namespace caffe {

template <typename Dtype>
void ReverseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int gap_per_T = N * feat_len;
  for (int j = 0; j < N; ++j) {
    const Dtype * cont_seq = bottom[0]->cpu_data() + j;
    const Dtype * feature_seq = bottom[1]->gpu_data() + j * feat_len;
    Dtype * feature_seq_reverse = top[0]->mutable_gpu_data() + j * feat_len;
    int cont_end = 0;
    int tt = -1;
    while (cont_end < T && *cont_seq != 0) {
      // get next sequence's length
      tt = -*cont_seq;
      int start = cont_end;
      do {
    	++cont_end;
    	cont_seq += N;
      } while ( cont_end < T && *cont_seq == 1 );
      CHECK_EQ(tt, cont_end-start) << "sequence length should be equal";
      const Dtype *feature_seq_end = feature_seq + tt * gap_per_T;
      for(int l=start; l < cont_end; ++l){
        feature_seq_end -= gap_per_T;
        caffe_copy( feat_len, feature_seq_end, feature_seq_reverse);
        feature_seq_reverse += gap_per_T;
      }
      feature_seq += tt*gap_per_T;
    }
  }
}

template <typename Dtype>
void ReverseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[1]) return;
  caffe_gpu_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_gpu_diff());
  const int gap_per_T = N * feat_len;
  for (int j = 0; j < N; ++j) {
    const Dtype * cont_seq = bottom[0]->cpu_data() + j;
    Dtype * feature_seq = bottom[1]->mutable_gpu_diff() + j * feat_len;
    const Dtype * feature_seq_reverse = top[0]->gpu_diff() + j * feat_len;
    int cont_end = 0;
    int tt = -1;
    while (cont_end < T && *cont_seq != 0) {
      // get next sequence's length
      tt = -*cont_seq;
      int start = cont_end;
      do {
    	++cont_end;
    	cont_seq += N;
      } while ( cont_end < T && *cont_seq == 1 );
      CHECK_EQ(tt, cont_end-start) << "sequence length should be equal";
      Dtype *feature_seq_end = feature_seq + tt * gap_per_T;
      for(int l=start; l < cont_end; ++l){
        feature_seq_end -= gap_per_T;
        caffe_copy( feat_len, feature_seq_reverse, feature_seq_end);
        feature_seq_reverse += gap_per_T;
      }
      feature_seq += tt*gap_per_T;
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ReverseLayer);

}  // namespace caffe
