#include <vector>

//#include "caffe/layer.hpp"
//#include "caffe/util/math_functions.hpp"
#include "caffe/common_layers.hpp"

namespace caffe {

template <typename Dtype>
void LabelConverterLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_class_ = this->layer_param_.label_converter_param().num_class();
  LOG(INFO) << "num_class = " << num_class_;

  CHECK(1 == bottom.size());
  vector<int> bottom_shape = bottom[0]->shape();
  CHECK(1 == bottom_shape[1]);
  CHECK(1 == bottom_shape[2]);
  CHECK(1 == bottom_shape[3]);
}

template <typename Dtype>
void LabelConverterLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape = bottom[0]->shape();

//  LOG(INFO) << "num_class in Reshape function  = " << this->layer_param_.label_converter_param().num_class();
  top_shape[1] = num_class_;

//  LOG(INFO) << "top_shape = " << top_shape[0] <<" "<< top_shape[1]<<" " << top_shape[2]<<" " << top_shape[3];

  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void LabelConverterLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num = top[0]->num(), chn = top[0]->channels(), height = top[0]->height(), width = top[0]->width();
  CHECK_EQ(chn, num_class_);
  CHECK_EQ(height, 1);
  CHECK_EQ(width, 1);

  const Dtype *pbottom = bottom[0]->cpu_data();
  Dtype *ptop = top[0]->mutable_cpu_data();

  for (int n = 0; n < num; ++n)
  {
    const Dtype label_multi = pbottom[n];
    CHECK(label_multi >= Dtype(0) && label_multi < Dtype(num_class_));
    for (int c = 0; c < chn; ++c)
    {
      if (label_multi == c)
        ptop[c] = Dtype(1);
      else
        ptop[c] = Dtype(0);
    }

    ptop += chn;
  }
}

template <typename Dtype>
void LabelConverterLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL) << "LabelConverterLayer should not have backward propagation";
}

#ifdef CPU_ONLY
STUB_GPU(LabelConverterLayer);
#endif

INSTANTIATE_CLASS(LabelConverterLayer);
REGISTER_LAYER_CLASS(LabelConverter);

}  // namespace caffe
