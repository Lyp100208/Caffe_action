#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "caffe/common.hpp"
#include "caffe/mpitask.hpp"
DEFINE_int32(gpu, -1,
    "Run in GPU mode on given device ID.");

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

void RepeatChannel(const boost::shared_ptr<caffe::Layer<float> > layer, const int fold)
{
  vector<shared_ptr<Blob<float> > > &blobs = layer->blobs();

  for (int i = 0; i < blobs.size(); i++)
  {
    shared_ptr<Blob<float> > &blob = blobs[i];
    vector<int> shape_old = blob->shape();
    if (shape_old.size() != 4)
      continue;
    const int chn_ori = shape_old[1];

    vector<int> shape_new = shape_old;
    shape_new[1] *= fold;
    Blob<float> rep(shape_new);

    LOG(INFO) << "begin to do blob repeating";
    LOG(INFO) << "size(blob[" << i << "] = " << shape_old[0] << ", " <<  shape_old[1] << ", " << shape_old[2] << ", " << shape_old[3];
    for (int n = 0; n < shape_old[0]; n++) {
      const float *pdata_src = blob->cpu_data() + blob->offset(n);
      float *pdata_new = rep.mutable_cpu_data() + rep.offset(n);
      for (int k = 0; k < fold; k++)
        caffe_copy(blob->count(1), pdata_src, pdata_new + rep.offset(0, chn_ori * k));
    }
    caffe_scal(rep.count(), 1.f / fold, rep.mutable_cpu_data());

    LOG(INFO) << "finished blob repeating";
    blobs[i]->CopyFrom(rep, false, true);
  }
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel name_layer num_fold"
              << " new_network.caffemodel" << std::endl;
    return 1;
  }
  
  const string model_file(argv[1]);
  const string trained_file(argv[2]);
  const string name_layer(argv[3]);
  const int fold = atoi(argv[4]);
  printf("repeat %d folds\n", fold);
  const string trained_file_new(argv[5]);

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
  if(FLAGS_gpu == -1) FLAGS_gpu = 0;
  thread_per_node = gpus_per_node - FLAGS_gpu;
  FLAGS_gpu += thread_id % thread_per_node;
  CUDA_CHECK(cudaSetDevice(FLAGS_gpu));
  node_num = thread_num / thread_per_node;
  Caffe::setThreadId(thread_id);
  Caffe::setThreadNum(thread_num); 
  Caffe::setThreadPerNode(thread_per_node); 
  Caffe::setNodeNum(node_num); 
  //To make sure the mpi driver is fine;
  float *test; int test_size=1024;
  cudaMalloc(&test,test_size*sizeof(float)); cudaMemset(test,1,test_size*sizeof(float));
  MPI_Allreduce(MPI_IN_PLACE, test, test_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  int error = 0;

  //load model
  Net<float> net(model_file, caffe::TRAIN);
  net.CopyTrainedLayersFrom(trained_file);

  //copy conv layer
  if (!net.has_layer(name_layer))
  {
     printf("[ERROR]: the net does not contain layer: %s\n", name_layer.c_str());
     exit(-1);    
  }
  const boost::shared_ptr<caffe::Layer<float> > layer_target = net.layer_by_name(name_layer);
  if (string(layer_target->type()) != string("Convolution"))
  {
     printf("[ERROR]: the layer to be repeated should be type of convolution\n");
     exit(-1);
  }  

  //revise conv layer
  RepeatChannel(layer_target, fold);

  //save model
  caffe::NetParameter net_param;
  net.ToProto(&net_param, false);
  WriteProtoToBinaryFile(net_param, trained_file_new.c_str());

  MPI_Finalize();
  return error;
}
