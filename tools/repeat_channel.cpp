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

  for (int n = 0; n < blobs.size(); n++)
  {
    vector<int> shape_old = blobs[n]->shape();
    if (shape_old.size() == 4)
    {
      const int num = shape_old[0], chn_ori = shape_old[1], hei = shape_old[2], wid = shape_old[3];
      printf("size(blob_old) = [%d,%d,%d,%d]\n", num, chn_ori, hei, wid);
      printf("size(blob_new) = [%d,%d,%d,%d]\n", num, fold, hei, wid);

      //get mean
      const float *pdata_src = blobs[n]->cpu_data();
      Blob<float> mean(num, 1, hei, wid);      
      float *pdata_mean = mean.mutable_cpu_data();
      memset(pdata_mean, 0, sizeof(float) * num * hei * wid);
      for (int nn = 0; nn < num; nn++)
      {
        for (int c = 0; c < chn_ori; c++)
        {
          for (int i = 0; i < hei * wid; i++)
            pdata_mean[i] += pdata_src[i] / chn_ori;

          pdata_src += hei * wid;
        }
        pdata_mean += hei * wid;
      }

      //repmat
      pdata_mean = mean.mutable_cpu_data();
      Blob<float> rep(num, fold, hei, wid);
      float *pdata_rep = rep.mutable_cpu_data();
      for (int nn = 0; nn < num; nn++)
      {
        for (int c = 0; c < fold; c++)
        {
          for (int i = 0; i < hei * wid; i++)
            pdata_rep[i] = pdata_mean[i];

          pdata_rep += hei * wid;
        }
        pdata_mean += hei * wid;
      }

      blobs[n]->CopyFrom(rep, false, true);
    }

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
