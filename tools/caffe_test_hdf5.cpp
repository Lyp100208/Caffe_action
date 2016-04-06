#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.hpp"
#include "caffe/common.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;
using std::string;
using std::vector;
using caffe::CPUTimer;

DEFINE_int32(gpu, -1,
    "Run in GPU mode on given device ID.");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file..");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning. "
    "Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 0,
    "The number of iterations to run.");
DEFINE_string(sigusr1_effect, "snapshot",
    "Optional; action to take when a SIGUSR1 signal is received: "
    "snapshot, stop, lr or none.");
DEFINE_string(sigusr2_effect, "lr",
    "Optional; action to take when a SIGUSR2 signal is received: "
    "snapshot, stop, lr or none.");
DEFINE_string(output, "prob",
    "output blob name");
DEFINE_string(folder, "",
    "folder to save output blob");

// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Device Query: show diagnostic information for a GPU device.
int device_query() {
  CHECK_GT(FLAGS_gpu, -1) << "Need a device ID to query.";
  LOG(INFO) << "Querying device ID = " << FLAGS_gpu;
  caffe::Caffe::SetDevice(FLAGS_gpu);
  caffe::Caffe::DeviceQuery();
  return 0;
}
RegisterBrewFunction(device_query);

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
  std::vector<std::string> model_names;
  boost::split(model_names, model_list, boost::is_any_of(",") );
  for (int i = 0; i < model_names.size(); ++i) {
    LOG(INFO) << "Finetuning from " << model_names[i];
    solver->net()->CopyTrainedLayersFrom(model_names[i]);
    for (int j = 0; j < solver->test_nets().size(); ++j) {
      solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
    }
  }
}

// Translate the signal effect the user specified on the command-line to the
// corresponding enumeration.
caffe::SolverAction::Enum GetRequestedAction(const std::string& flag_value) {
  if (flag_value == "stop") {
    return caffe::SolverAction::STOP;
  }
  if (flag_value == "snapshot") {
    return caffe::SolverAction::SNAPSHOT;
  }
  if (flag_value == "none") {
    return caffe::SolverAction::NONE;
  }
  if (flag_value == "lr") {
    return caffe::SolverAction::LR;
  }
  LOG(FATAL) << "Invalid signal effect \"" << flag_value << "\" was specified";
}

// Test: score a model.
int test() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";
  CHECK_GT(FLAGS_output.size(), 0) << "Need output blob name";
  CHECK_GT(FLAGS_folder.size(), 0) << "Need output folder";

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);

  //get batch_size
  int batch_size;
  const vector<shared_ptr<Layer<float> > > &layers = caffe_net.layers();
  if (layers[0].get()->layer_param().has_hdf5_data_param()) {
    batch_size = layers[0].get()->layer_param().hdf5_data_param().batch_size();
  } else {
    LOG(FATAL) << "type of first layer: " << layers[0]->type() << " is not hdf5 layer";
  }
  // Read test_list
  //const int id_thread = Caffe::getThreadId();
  const int num_thread = Caffe::getThreadNum();
  if (num_thread != 1) {
    LOG(ERROR) << "Do not support multi-gpu in hdf5 testing";
  }
  if (0 == FLAGS_iterations) {
    LOG(ERROR) << "iteration should be set";
  }
  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  vector<Blob<float>* > bottom_vec;
  char path_file[260];
  for (int i = 0; i < FLAGS_iterations; ++i) {
    LOG(INFO) << i+1 << "/" << FLAGS_iterations;
    float iter_loss;
    CPUTimer timer;
    timer.Start();
    caffe_net.Forward(bottom_vec, &iter_loss);
    LOG(INFO) << "forward cost " << timer.MilliSeconds() << " ms";

    //save softmax output
    const shared_ptr<Blob<float> > blob_output = caffe_net.blob_by_name(FLAGS_output);
    CHECK_EQ(blob_output->num(), batch_size);
    const int channels = blob_output->channels();
    const int height = blob_output->height();
    const int width = blob_output->width();

    const float* softmax_cpu_data = blob_output->cpu_data();
    sprintf(path_file, "%s/batch_%d.bin", FLAGS_folder.c_str(), i);
    FILE  *fid = fopen(path_file, "wb");
    CHECK(fid != NULL) << "faied to open file " << path_file;
    
    fwrite(&(width), sizeof(int), 1, fid);
    fwrite(&(height), sizeof(int), 1, fid);
    fwrite(&(channels), sizeof(int), 1, fid);
    fwrite(&(batch_size), sizeof(int), 1, fid);
    fwrite(softmax_cpu_data, sizeof(float), width * height * channels * batch_size, fid);
    fclose(fid);
  }

  return 0;
}
RegisterBrewFunction(test);

int Caffe_MPI_Init(int argc, char** argv){
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if(provided != MPI_THREAD_MULTIPLE)
  {
     LOG(INFO)<<"MPI do not Support Multiple thread\n";
     exit(0);
  }
  int thread_id=0,thread_num=0,gpus_per_node=0,node_num=0;
  int color = 0, key = 0, count = 0, name_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &thread_id);
  MPI_Comm_size(MPI_COMM_WORLD, &thread_num);

  char node_name[MPI::MAX_PROCESSOR_NAME];
  MPI::Get_processor_name(node_name, name_size);
  char *all_node_name = (char *)malloc(sizeof(char) * MPI::MAX_PROCESSOR_NAME * thread_num);
  MPI_Allgather(node_name,MPI::MAX_PROCESSOR_NAME,MPI_CHAR,all_node_name,
      MPI::MAX_PROCESSOR_NAME,MPI_CHAR,MPI_COMM_WORLD); 
  MPI_Comm dup_comm, sub_comm;
  for(int i = 0; i < thread_num; i++)
  {
    if (i == 0) node_num = 1;
    else if(strcmp(node_name,all_node_name + MPI::MAX_PROCESSOR_NAME * (i - 1)) != 0){
      node_num++;
    }
    if(strcmp(node_name,all_node_name + MPI::MAX_PROCESSOR_NAME * i) == 0)
    {  
       if(count == 0)
           color = i;
       if(i == thread_id)  
       {
         key = count;  
         break;  
       }
       count++;
    }
  }
  MPI_Comm_dup(MPI_COMM_WORLD,&dup_comm);
  MPI_Comm_split(dup_comm,color,key,&sub_comm);

  int sub_thread_num = 0, min_sub_thread_id = 0;
  MPI_Comm_size(sub_comm, &sub_thread_num);
  MPI_Allreduce(&thread_id, &min_sub_thread_id, sizeof(int), MPI_INT, MPI_MIN, sub_comm);

  LOG(INFO)<<"min sub thread id:"<<min_sub_thread_id<<" node name:"<<node_name;

  if(FLAGS_gpu == -1) FLAGS_gpu = 0;

  CUDA_CHECK(cudaGetDeviceCount(&gpus_per_node));
  if(FLAGS_gpu >=0 && FLAGS_gpu <= gpus_per_node) {
    FLAGS_gpu += (thread_id - min_sub_thread_id);
    LOG(INFO)<<"FLAGS_gpu:"<<FLAGS_gpu;
    CUDA_CHECK(cudaSetDevice(FLAGS_gpu));
  }
  else
    FLAGS_gpu = -1;
  Caffe::setThreadId(thread_id);
  Caffe::setThreadNum(thread_num); 
  Caffe::setThreadPerNode(sub_thread_num); 
  Caffe::setNodeNum(node_num); 
  return 0; 
}

int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: caffe <command> <args>\n\n"
      "commands:\n"
      "  test            score a model\n"
      "  device_query    show GPU diagnostic information\n");
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);
  Caffe_MPI_Init(argc, argv);
  
  int error = GetBrewFunction(caffe::string("test"))();
  MPI_Finalize();
  return error;
}
