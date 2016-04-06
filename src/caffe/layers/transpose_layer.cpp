#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TransposeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
    const vector<Blob<Dtype>*>& top) {
  // check
  TransposeParameter param = this->layer_param_.transpose_param();
  CHECK_EQ(param.dim_size(), 2)
  << "\n need 2 dimensions to apply transpose";
  CHECK_NE(param.dim(0), param.dim(1))
  << "\n need 2 dimensions to apply transpose";
  
  // assign
  dim_1 = param.dim(0);
  dim_2 = param.dim(1);
  if(param.dim(0)>param.dim(1)) {
      dim_1=param.dim(1);
      dim_2=param.dim(0);
  }
  else {
      dim_2=param.dim(1);
      dim_1=param.dim(0);
  }
}

template <typename Dtype>
void TransposeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const  vector<Blob<Dtype>*>& top) {
  // check
  CHECK_LT(dim_1, bottom[0]->shape().size());
  CHECK_LT(dim_2, bottom[0]->shape().size());
 
  vector<int> shape=bottom[0]->shape();
  int temp=shape[dim_2];
  shape[dim_2]=shape[dim_1];
  shape[dim_1]=temp;
  top[0]->Reshape(shape);
}

template <typename Dtype>
void TransposeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void TransposeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(TransposeLayer);
#endif

INSTANTIATE_CLASS(TransposeLayer);
REGISTER_LAYER_CLASS(Transpose);
}  // namespace caffe
