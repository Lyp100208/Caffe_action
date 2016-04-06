#include <glog/logging.h>
 
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.hpp"
#include "caffe/common.hpp"
#include "caffe/util/io.hpp"
using namespace caffe;
//#include "caffe/mpitask.hpp"

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
  LOG(INFO) << "In caffe_test_ms_mv test()";
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

  //get batch_size, num_viewss and num_seg
  const vector<shared_ptr<Layer<float> > > &layers = caffe_net.layers();
  VideoDataParameter para;
  if (layers[0].get()->layer_param().has_video_data_param())
    para = layers[0].get()->layer_param().video_data_param();
  else
    LOG(FATAL) << "Layer " << layers[0]->type() << " do not contains video_data_param";

  const string vidlist = para.source();
  const int num_views = para.num_views();
  const int num_seg = para.num_segments();
  const int num_pervid = num_views * num_seg;
  int batch_size = para.batch_size();
  const int id_thread = Caffe::getThreadId();
  const int num_thread = Caffe::getThreadNum();
  CHECK_EQ(0, batch_size % num_thread);
  batch_size /= num_thread;

  // Read vidlist
  vector<string> names_vid;
  vector<int> lens_vid, labels_gt;
  ReadVidList(vidlist, names_vid, lens_vid, labels_gt);
  const int num_vid = int(names_vid.size());
  LOG(INFO) << "get " << num_vid << " videos in caffe_test_multiseg_multiview.cpp";
  if (0 == FLAGS_iterations)
    FLAGS_iterations = ceil(float(num_vid - num_vid / num_thread * (num_thread - 1)) / batch_size);

  vector<Blob<float>* > bottom_vec;
  const int start_in_list = num_vid / num_thread * id_thread;
  for (int i = 0; i < FLAGS_iterations; ++i) {
    LOG(INFO) << i+1 << "/" << FLAGS_iterations;
    float iter_loss;
    caffe_net.Forward(bottom_vec, &iter_loss);

    //save softmax output
    const shared_ptr<Blob<float> > blob_output = caffe_net.blob_by_name(FLAGS_output);
//    CHECK_EQ(blob_output->num(), batch_size * num_pervid);
//    const int channels = blob_output->channels();
//    const int height = blob_output->height();
//    const int width = blob_output->width();
    CHECK_EQ(0, blob_output->count() % (num_pervid * batch_size));
    const int chn = blob_output->count() / (num_pervid * batch_size);

    const float* poutput = blob_output->cpu_data();
    for (int j = 0; j < batch_size; j++) {
      const int id_vid = start_in_list + batch_size * i + j;
      if (id_vid >= num_vid)
        break;
      
      const string path_file = FLAGS_folder + "/" + names_vid[id_vid] + "_"
         + std::to_string(num_seg) + "segs_" + std::to_string(num_views) + "views" + ".bin";

      SaveBinFile(poutput, num_pervid, chn, 1, 1, path_file);
      poutput += chn * num_pervid;
    }
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
  LOG(INFO) << "In caffe_test_ms_mv main";
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
