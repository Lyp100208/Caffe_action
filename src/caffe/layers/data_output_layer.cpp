#include <vector>

#include "caffe/layer.hpp"
#include "caffe/data_layers.hpp"

using namespace cv;

namespace caffe {
  
/**
* @brief @f$
*
*/

template <typename Dtype>
void DataOutputLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  DataOutputParameter data_output_param = this->layer_param_.data_output_param();
  
  // check
  CHECK_EQ(data_output_param.file_name_size(), bottom.size())
  << "\n each bottom need a file name to output";
  
  // assign
  for(size_t i=0;i<bottom.size();i++) {
    file_name.push_back(data_output_param.file_name(i));
  }
  output_type=data_output_param.type();
  bottom_num.resize(bottom.size());
  bottom_num_stride.resize(bottom.size());
}

template <typename Dtype>
void DataOutputLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  // assign
  for(size_t i=0;i<bottom.size();i++) {
    bottom_num[i]=bottom[i]->num();
    bottom_num_stride[i]=bottom[i]->count(1);
  }
}

template <typename Dtype>
void DataOutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{
  if(Caffe::getThreadId()!=0) {
    return;
  }
  
  // output
  for(size_t i=0;i<file_name.size();i++) {
    const Dtype* bottom_data=bottom[i]->cpu_data();
    
    if(output_type==DataOutputParameter_FileType_TEXT) {
      FILE *fp_output=fopen(file_name[i].c_str(), "a");
      for(size_t j=0;j<bottom_num[i];j++) {
        for(size_t k=0;k<bottom_num_stride[i];k++) {
          fprintf(fp_output, "%f ", *bottom_data);
          bottom_data++;
        }
        fprintf(fp_output, "\n");
      }
      fclose(fp_output);
    }
    else {
      FILE *fp_output=fopen(file_name[i].c_str(), "ab");
      fwrite(bottom_data, sizeof(Dtype), bottom[i]->count(0), fp_output);
      fclose(fp_output);
    }
  }  
}

INSTANTIATE_CLASS(DataOutputLayer);
REGISTER_LAYER_CLASS(DataOutput);

}  // namespace caffe
