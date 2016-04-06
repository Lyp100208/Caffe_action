#include <glog/logging.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <fstream>
//#include "boost/algorithm/string.hpp"
#include "caffe/mpitask.hpp"
#include "caffe/caffe.hpp"

using namespace caffe;

int id_by_name(const string &query, const vector<string> &names)
{
  int i = 0;
  for (; i < int(names.size()); ++i)
  {
    if (query == names[i])
      return i;
  }
  return -1;
}

string trim(string& s)
{
  if (s.empty()) return s;
  s.erase(0,s.find_first_not_of("\n"));
  s.erase(s.find_last_not_of("\n") + 1);

  s.erase(0,s.find_first_not_of("\r"));
  s.erase(s.find_last_not_of("\r") + 1);

  s.erase(0,s.find_first_not_of("\t"));
  s.erase(s.find_last_not_of("\t") + 1);

  s.erase(0,s.find_first_not_of(" "));
  s.erase(s.find_last_not_of(" ") + 1);
  return s;
}

int main(int argc ,char **argv) {
  
  const string deploy = argv[1];
  const string model = argv[2];
  const string test_list = argv[3];
  const string saved_folder = argv[4];
  const string softmax_name = argv[5];
  int gpu = atoi(argv[6]);
  // Set device id and mode
  Caffe::SetDevice(gpu);
  Caffe::set_mode(Caffe::GPU);

  // Read test_list
   vector<string> video_names;
  std::ifstream infile(test_list.c_str());
  char str[100];
  LOG(INFO) << "Begin to read list";
  while(infile.getline(str,sizeof(str))){
    string line(str);
    size_t index = line.find_first_of(" ");
    string video_name =  line.substr(0,index);
//    LOG(INFO) << trim(video_name);
    video_names.push_back(trim(video_name));
  }
  infile.close();

  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if(provided != MPI_THREAD_MULTIPLE)
  {
     LOG(INFO)<<"MPI do not Support Multiple thread\n";
     exit(0);
  }
  int thread_id=0,thread_num=0,gpus_per_node=0,thread_per_node=0,node_num=0;
  MPI_Comm_rank(MPI_COMM_WORLD, &thread_id); 
  MPI_Comm_size(MPI_COMM_WORLD, &thread_num);
  CUDA_CHECK(cudaGetDeviceCount(&gpus_per_node));
  thread_per_node = gpus_per_node - gpu;
  gpu += thread_id % thread_per_node;
  CUDA_CHECK(cudaSetDevice(gpu));
  node_num = thread_num / thread_per_node;
  Caffe::setThreadId(thread_id);
  Caffe::setThreadNum(thread_num); 
  Caffe::setThreadPerNode(thread_per_node); 
  Caffe::setNodeNum(node_num); 

  MpiTaskList<float>* task_list = new MpiTaskList<float>();
  Caffe::setTaskList(task_list);
  task_list->StartInternalThread();
  // Instantiate the caffe net.
  Net<float> caffe_net(deploy, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(model);

  LOG(INFO) << "Finished instanitiate net!";
  const vector<shared_ptr<Layer<float> > > &layers = caffe_net.layers();

  int num_seg, num_view;
  int batch_size;
  LOG(INFO) << "type of first layer: " << layers[0]->type();

  if (layers[0].get()->layer_param().has_video_data_param()){
    num_seg = layers[0].get()->layer_param().video_data_param().num_segments();
    num_view = layers[0].get()->layer_param().video_data_param().num_views();
    batch_size = layers[0].get()->layer_param().video_data_param().batch_size();
  }
  else if(layers[0].get()->layer_param().has_video2_data_param()){
    num_seg = layers[0].get()->layer_param().video2_data_param().num_segments();
    num_view = layers[0].get()->layer_param().video2_data_param().num_views();
    batch_size = layers[0].get()->layer_param().video2_data_param().batch_size();
  }else{
    LOG(INFO) << "First layer is not data layer ! ";
    return -1;
  }
  CHECK((batch_size % (num_seg*num_view)) == 0);
  int batch_video = batch_size / num_seg / num_view;
  LOG(INFO) << "batch_size = " << batch_size << " num_segments = " << num_seg << " num_view = " << num_view << " #video in batch = " << batch_video;
  int iterations = int(float(video_names.size()) / batch_video);
  if (iterations * batch_video < int(video_names.size()))
    iterations++;

  int num_class =  caffe_net.blob_by_name(softmax_name).get()->count() / batch_size;
  LOG(INFO) << "num_class :" << num_class;
  LOG(INFO) << "Running for " << iterations << " iterations."; 
  LOG(INFO) << "use gpu " << gpu << " for testing";
  LOG(INFO) << "thread_id = " << thread_id;
  LOG(INFO) << "thread_num = " << thread_num;
  LOG(INFO) << "thread_per_node = " << thread_per_node;
  vector<Blob<float>* > bottom_vec;
  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < iterations; ++i) {
    float iter_loss;
    const vector<Blob<float>*>& result = caffe_net.Forward(bottom_vec, &iter_loss);
    LOG(INFO) << "finished forward";
    //save softmax output
    const shared_ptr<Blob<float> > blob_softmax = caffe_net.blob_by_name(softmax_name);
    CHECK(blob_softmax->num() == batch_size);
    const int channels = blob_softmax->channels();
    const int height = blob_softmax->height();
    const int width = blob_softmax->width();

    const float* softmax_cpu_data = blob_softmax->cpu_data();
    for ( int j = 0; j < batch_video; ++j){
      if (batch_video * i + j >= int(video_names.size()))
         break;

        const string path_file = saved_folder + "/" + video_names[i * (batch_video) + j] + ".bin";
        FILE  *fid = fopen(path_file.c_str(), "wb");
        CHECK(fid != NULL) << "faied to open file " << path_file;
        LOG(INFO) << "Begin to write "<< path_file;
        
        fwrite(&(width), sizeof(int), 1, fid);
        fwrite(&(height), sizeof(int), 1, fid);
        fwrite(&(channels), sizeof(int), 1, fid);
        const int num_tmp = num_seg * num_view;
        fwrite(&(num_tmp), sizeof(int), 1, fid);
        fwrite(softmax_cpu_data, sizeof(float), width * height * channels * num_seg * num_view, fid);
        fclose(fid);

        softmax_cpu_data += width * height * channels * num_seg * num_view;
    }
    
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        const std::string& output_name = caffe_net.blob_names()[
            caffe_net.output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
  }
  loss /= iterations;
  LOG(INFO) << "Loss: " << loss;
  
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net.blob_names()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight = caffe_net.blob_loss_weights()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }
  MPI_Finalize();
  return 0;
}
