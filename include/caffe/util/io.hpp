#ifndef CAFFE_UTIL_IO_H_
#define CAFFE_UTIL_IO_H_

#include <unistd.h>
#include <string>
#include <opencv2/opencv.hpp>

#include "google/protobuf/message.h"
#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/rng.hpp"

#define HDF5_NUM_DIMS 4

namespace caffe {

using ::google::protobuf::Message;

inline void MakeTempFilename(string* temp_filename) {
  temp_filename->clear();
  *temp_filename = "/tmp/caffe_test.XXXXXX";
  char* temp_filename_cstr = new char[temp_filename->size() + 1];
  // NOLINT_NEXT_LINE(runtime/printf)
  strcpy(temp_filename_cstr, temp_filename->c_str());
  int fd = mkstemp(temp_filename_cstr);
  CHECK_GE(fd, 0) << "Failed to open a temporary file at: " << *temp_filename;
  close(fd);
  *temp_filename = temp_filename_cstr;
  delete[] temp_filename_cstr;
}

inline void MakeTempDir(string* temp_dirname) {
  temp_dirname->clear();
  *temp_dirname = "/tmp/caffe_test.XXXXXX";
  char* temp_dirname_cstr = new char[temp_dirname->size() + 1];
  // NOLINT_NEXT_LINE(runtime/printf)
  strcpy(temp_dirname_cstr, temp_dirname->c_str());
  char* mkdtemp_result = mkdtemp(temp_dirname_cstr);
  CHECK(mkdtemp_result != NULL)
      << "Failed to create a temporary directory at: " << *temp_dirname;
  *temp_dirname = temp_dirname_cstr;
  delete[] temp_dirname_cstr;
}

bool ReadProtoFromTextFile(const char* filename, Message* proto);

inline bool ReadProtoFromTextFile(const string& filename, Message* proto) {
  return ReadProtoFromTextFile(filename.c_str(), proto);
}

inline void ReadProtoFromTextFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromTextFile(filename, proto));
}

inline void ReadProtoFromTextFileOrDie(const string& filename, Message* proto) {
  ReadProtoFromTextFileOrDie(filename.c_str(), proto);
}

void WriteProtoToTextFile(const Message& proto, const char* filename);
inline void WriteProtoToTextFile(const Message& proto, const string& filename) {
  WriteProtoToTextFile(proto, filename.c_str());
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto);

inline bool ReadProtoFromBinaryFile(const string& filename, Message* proto) {
  return ReadProtoFromBinaryFile(filename.c_str(), proto);
}

inline void ReadProtoFromBinaryFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromBinaryFile(filename, proto));
}

inline void ReadProtoFromBinaryFileOrDie(const string& filename,
                                         Message* proto) {
  ReadProtoFromBinaryFileOrDie(filename.c_str(), proto);
}


void WriteProtoToBinaryFile(const Message& proto, const char* filename);
inline void WriteProtoToBinaryFile(
    const Message& proto, const string& filename) {
  WriteProtoToBinaryFile(proto, filename.c_str());
}

bool ReadFileToDatum(const string& filename, const int label, Datum* datum);

inline bool ReadFileToDatum(const string& filename, Datum* datum) {
  return ReadFileToDatum(filename, -1, datum);
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum);

inline bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color, Datum* datum) {
  return ReadImageToDatum(filename, label, height, width, is_color,
                          "", datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, Datum* datum) {
  return ReadImageToDatum(filename, label, height, width, true, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    const bool is_color, Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, is_color, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, true, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    const std::string & encoding, Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, true, encoding, datum);
}

bool DecodeDatumNative(Datum* datum);
bool DecodeDatum(Datum* datum, bool is_color);

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color);

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width);

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color);

cv::Mat ReadImageToCVMat(const string& filename);

bool ReadSegmentFlowToDatum(const string& filename, const int label,
        const vector<int> offsets, const int height, const int width,
        const int length, Datum* datum, const bool is_color,
        const int len_vid = 0);

bool ReadSegmentRGBToDatum(const string& filename, const int label,
    const vector<int> offsets, const int height, const int width,
    const int length, Datum* datum,
    bool is_color, const int len_vid = 0, const string& suffix = "jpg");

bool ReadSegmentRGBToDatum(const string& filename, const int label,
    const vector<int> &offsets, const vector<vector<cv::Rect> > &bboxes,
    const int new_height, const int new_width, const int new_length,
    Datum* datum, bool is_color,
    const int crop_width, const int crop_height,
    const string& suffix = "jpg");

bool ReadSegmentFlowToDatum(const string& filename, const int label,
    const vector<int> &offsets, const vector<vector<cv::Rect> > &bboxes,
    const int new_height, const int new_width, const int new_length,
    Datum* datum, const bool is_color, const int crop_width, const int crop_height);

cv::Mat DecodeDatumToCVMatNative(const Datum& datum);
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color);

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum);

template <typename Dtype>
void hdf5_load_nd_dataset_helper(
    hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
    Blob<Dtype>* blob);

template <typename Dtype>
void hdf5_load_nd_dataset(
    hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
    Blob<Dtype>* blob);

template <typename Dtype>
void hdf5_save_nd_dataset(
    const hid_t file_id, const string& dataset_name, const Blob<Dtype>& blob);

template <typename Dtype>
void LoadBinFile(const string &path_file, int &pos, vector<Dtype> &data, vector<int> &sizes);

template <typename Dtype>
void LoadBinFile(const string &path_file, vector<Dtype> &data, vector<int> &shape);

int LoadBinFile(const string &path_file, vector<int> &sizes);

template <typename Dtype>
void SaveBinFile(const Dtype *pdata, const vector<int> sizes, const string &path_file);

template<typename Dtype>
void SaveBinFile(const Dtype *pdata, const int num, const int chn, const int height, const int width, const string &path_file);

void Linspace(const int s, const int e, const int num, vector<int> &ids);

void Cirspace(const int len, const int num, vector<int> &ids);

void GenSmpIds(const int len, const int new_length, const int stride_tmp, vector<int> &starts);

void ReadVidList(const string &vidlist, vector<string> &names, vector<int> &lengths, vector<int> &labels);

void SaveVidList(const string &vidlist, const vector<string> &names, const vector<int> &lengths, const vector<int> &labels);

void SampleSegments(const int len_vid, const int seq_length, const int stride_seq, const bool is_train, const bool do_rolling,
    const int num_segments, vector<int> &starts, vector<int> &size_seq, caffe::rng_t* rng = NULL);

void ReadClassNames(const string &path_name_list, vector<string> &names_class);

void SaveTube(const string &tube_path, const vector<vector<int> > &id_seq, const vector<vector<float> > &score_seq, const vector<float> &score_tube);

void LoadTube(const string &tube_path, vector<vector<int> > &id_seq, vector<vector<float> > &score_seq, vector<float> &score_tube);

void DenseSampleBBox(const int width, const int height, const float maxAspectRatio, const float olpstep, const float minAreaRatio, std::vector<cv::Rect> &boxes, std::vector<int> &cumcnt);

int Sample_Sto(const vector<float>& score);
}  // namespace caffe


#endif   // CAFFE_UTIL_IO_H_
