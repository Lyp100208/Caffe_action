#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

namespace caffe {

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  if (height > 0 && width > 0) {
    int new_width = width;
    int new_height = height;
    if (height == 1 || width == 1) {
      float length = height > width ? height : width;
      if (cv_img_origin.rows < cv_img_origin.cols) { // height < width
        float scale = length / cv_img_origin.rows;
        new_width = scale * cv_img_origin.cols;
        new_height = length;
      }
      else { // width <= height
        float scale = length / cv_img_origin.cols;
        new_width = length;
        new_height = scale * cv_img_origin.rows;
      }
    }
    cv::resize(cv_img_origin, cv_img, cv::Size(new_width, new_height));
  } else {
    cv_img = cv_img_origin;
  }
  return cv_img;
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width) {
  return ReadImageToCVMat(filename, height, width, true);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color) {
  return ReadImageToCVMat(filename, 0, 0, is_color);
}

cv::Mat ReadImageToCVMat(const string& filename) {
  return ReadImageToCVMat(filename, 0, 0, true);
}
// Do the file extension and encoding match?
static bool matchExt(const std::string & fn,
                     std::string en) {
  size_t p = fn.rfind('.');
  std::string ext = p != fn.npos ? fn.substr(p) : fn;
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  std::transform(en.begin(), en.end(), en.begin(), ::tolower);
  if ( ext == en )
    return true;
  if ( en == "jpg" && ext == "jpeg" )
    return true;
  return false;
}
bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum) {
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);
  if (cv_img.data) {
    if (encoding.size()) {
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&
          matchExt(filename, encoding) )
        return ReadFileToDatum(filename, label, datum);
      std::vector<uchar> buf;
      cv::imencode("."+encoding, cv_img, buf);
      datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                      buf.size()));
      datum->set_label(label);
      datum->set_encoded(true);
      return true;
    }
    CVMatToDatum(cv_img, datum);
    datum->set_label(label);
    return true;
  } else {
    return false;
  }
}

bool ReadFileToDatum(const string& filename, const int label,
    Datum* datum) {
  std::streampos size;

  fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    std::string buffer(size, ' ');
    file.seekg(0, ios::beg);
    file.read(&buffer[0], size);
    file.close();
    datum->set_data(buffer);
    datum->set_label(label);
    datum->set_encoded(true);
    return true;
  } else {
    return false;
  }
}

cv::Mat DecodeDatumToCVMatNative(const Datum& datum) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  cv_img = cv::imdecode(vec_data, -1);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv_img = cv::imdecode(vec_data, cv_read_flag);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}

// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum
// If Datum is not encoded will do nothing
bool DecodeDatumNative(Datum* datum) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMatNative((*datum));
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}
bool DecodeDatum(Datum* datum, bool is_color) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMat((*datum), is_color);
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  std::string buffer(datum_size, ' ');
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  datum->set_data(buffer);
}

// Verifies format of data stored in HDF5 file and reshapes blob accordingly.
template <typename Dtype>
void hdf5_load_nd_dataset_helper(
    hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
    Blob<Dtype>* blob) {
  // Verify that the dataset exists.
  CHECK(H5LTfind_dataset(file_id, dataset_name_))
      << "Failed to find HDF5 dataset " << dataset_name_;
  // Verify that the number of dimensions is in the accepted range.
  herr_t status;
  int ndims;
  status = H5LTget_dataset_ndims(file_id, dataset_name_, &ndims);
  CHECK_GE(status, 0) << "Failed to get dataset ndims for " << dataset_name_;
  CHECK_GE(ndims, min_dim);
  CHECK_LE(ndims, max_dim);

  // Verify that the data format is what we expect: float or double.
  std::vector<hsize_t> dims(ndims);
  H5T_class_t class_;
  status = H5LTget_dataset_info(
      file_id, dataset_name_, dims.data(), &class_, NULL);
  CHECK_GE(status, 0) << "Failed to get dataset info for " << dataset_name_;
  CHECK_EQ(class_, H5T_FLOAT) << "Expected float or double data";

  vector<int> blob_dims(dims.size());
  for (int i = 0; i < dims.size(); ++i) {
    blob_dims[i] = dims[i];
  }
  blob->Reshape(blob_dims);
}

template <>
void hdf5_load_nd_dataset<float>(hid_t file_id, const char* dataset_name_,
        int min_dim, int max_dim, Blob<float>* blob) {
  hdf5_load_nd_dataset_helper(file_id, dataset_name_, min_dim, max_dim, blob);
  herr_t status = H5LTread_dataset_float(
    file_id, dataset_name_, blob->mutable_cpu_data());
  CHECK_GE(status, 0) << "Failed to read float dataset " << dataset_name_;
}

template <>
void hdf5_load_nd_dataset<double>(hid_t file_id, const char* dataset_name_,
        int min_dim, int max_dim, Blob<double>* blob) {
  hdf5_load_nd_dataset_helper(file_id, dataset_name_, min_dim, max_dim, blob);
  herr_t status = H5LTread_dataset_double(
    file_id, dataset_name_, blob->mutable_cpu_data());
  CHECK_GE(status, 0) << "Failed to read double dataset " << dataset_name_;
}

template <>
void hdf5_save_nd_dataset<float>(
    const hid_t file_id, const string& dataset_name, const Blob<float>& blob) {
  hsize_t dims[HDF5_NUM_DIMS];
  dims[0] = blob.num();
  dims[1] = blob.channels();
  dims[2] = blob.height();
  dims[3] = blob.width();
  herr_t status = H5LTmake_dataset_float(
      file_id, dataset_name.c_str(), HDF5_NUM_DIMS, dims, blob.cpu_data());
  CHECK_GE(status, 0) << "Failed to make float dataset " << dataset_name;
}

template <>
void hdf5_save_nd_dataset<double>(
    const hid_t file_id, const string& dataset_name, const Blob<double>& blob) {
  hsize_t dims[HDF5_NUM_DIMS];
  dims[0] = blob.num();
  dims[1] = blob.channels();
  dims[2] = blob.height();
  dims[3] = blob.width();
  herr_t status = H5LTmake_dataset_double(
      file_id, dataset_name.c_str(), HDF5_NUM_DIMS, dims, blob.cpu_data());
  CHECK_GE(status, 0) << "Failed to make double dataset " << dataset_name;
}

void ReadImage2String(const cv::Mat &cv_img, string &datum_tmp, int &id)
{
  const bool is_color = cv_img.channels() == 3;
  if (!is_color)
    CHECK_EQ(cv_img.channels(), 1) << "image channel should be 1 or 3";

  const int hei = cv_img.rows, wid = cv_img.cols;
  if (is_color)
  {
    cv::Mat bgr[3];
    cv::split(cv_img, bgr);

    for (int c = 0; c < 3; c++)
    {
      cv::Mat &img = bgr[c];
      for (int h = 0; h < hei; ++h)
      {
        const uchar *prow = img.ptr<uchar>(h);
        for (int w = 0; w < wid; ++w)
          datum_tmp[id++] = static_cast<char>(prow[w]);
      }
    }
  }
  else
  {
    for (int h = 0; h < hei; ++h)
    {
      const uchar *prow = cv_img.ptr<uchar>(h);
      for (int w = 0; w < wid; ++w)
        datum_tmp[id++] = static_cast<char>(prow[w]);
    }
  }
}


bool ReadSegmentRGBToDatum(const string& filename, const int label,
    const vector<int> offsets, const int height, const int width,
    const int length, Datum* datum, bool is_color,
    const int len_vid, const string& suffix) {

  string datum_tmp;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

  int id = 0;
  char tmp[30];
  for (int i = 0; i < offsets.size(); ++i)
  {
    int offset = offsets[i];
    CHECK_GE(offset, 0);
    for (int file_id = 0; file_id < length; ++file_id)
    {
      if (len_vid)
        sprintf(tmp,"%d.",int(file_id+offset) % len_vid);
      else
        sprintf(tmp,"%d.",int(file_id+offset));

      string filename_t = filename + "/" + tmp + suffix;

      cv::Mat cv_img_origin = cv::imread(filename_t, cv_read_flag);
      if (!cv_img_origin.data){
        LOG(ERROR) << "Could not load file " << filename_t;
        return false;
      }

      cv::Mat cv_img;
      if (height > 0 && width > 0){
        cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
      }else{
        cv_img = cv_img_origin;
      }

      int num_channels = (is_color ? 3 : 1);
      const int hei = cv_img.rows, wid = cv_img.cols;
      if (file_id==0 && i==0)
      {
        datum->set_channels(num_channels*length*offsets.size());
        datum->set_height(hei);
        datum->set_width(wid);
        datum->set_label(label);
        datum->clear_data();
        datum->clear_float_data();
        datum_tmp.resize(hei * wid * datum->channels());
      }

      ReadImage2String(cv_img, datum_tmp, id);
    }
  }

  datum->set_data(datum_tmp);
  return true;
}

bool ReadSegmentFlowToDatum(const string& filename, const int label,
    const vector<int> offsets, const int height, const int width, const int length, Datum* datum, const bool is_color, const int len_vid){
  cv::Mat cv_img_x, cv_img_y;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
  int num_channels = is_color ? 3 : 1;

  int id = 0;
  char tmp[30];
  for (int i = 0; i < offsets.size(); ++i)
  {
    int offset = offsets[i];
    CHECK_GE(offset, 0);
    for (int file_id = 0; file_id < length; ++file_id)
    {
      if (len_vid)
        sprintf(tmp,"%d_x.png",int(file_id+offset) % len_vid);
      else
        sprintf(tmp,"%d_x.png",int(file_id+offset));

      string filename_x = filename + "/" + tmp;
      cv::Mat cv_img_origin_x = cv::imread(filename_x, cv_read_flag);
      if (len_vid)
        sprintf(tmp,"%d_y.png",int(file_id+offset) % len_vid);
      else
        sprintf(tmp,"%d_y.png",int(file_id+offset));

      string filename_y = filename + "/" + tmp;
      cv::Mat cv_img_origin_y = cv::imread(filename_y, cv_read_flag);
      if (!cv_img_origin_x.data || !cv_img_origin_y.data){
        LOG(ERROR) << "Could not load file " << filename_x << " or " << filename_y;
        return false;
      }
      if (height > 0 && width > 0 && (height != cv_img_origin_x.rows ||  width != cv_img_origin_x.cols)){
        cv::resize(cv_img_origin_x, cv_img_x, cv::Size(width, height));
        cv::resize(cv_img_origin_y, cv_img_y, cv::Size(width, height));
      }else{
        cv_img_x = cv_img_origin_x;
        cv_img_y = cv_img_origin_y;
      }

      const int hei = cv_img_x.rows, wid = cv_img_x.cols;
      if (file_id==0 && i==0)
      {
        datum->set_channels(num_channels * 2 * length * offsets.size());
        datum->set_height(hei);
        datum->set_width(wid);
        datum->set_label(label);
        datum->clear_data();
        datum->clear_float_data();
        datum->mutable_data()->resize(hei * wid * datum->channels());
      }

      string &datum_ref = *(datum->mutable_data());
      ReadImage2String(cv_img_x, datum_ref, id);
      ReadImage2String(cv_img_y, datum_ref, id);
    }
  }

  return true;
}

bool ReadSegmentRGBToDatum(const string& filename, const int label,
    const vector<int> &offsets, const vector<vector<cv::Rect> > &bboxes,
    const int new_height, const int new_width, const int new_length,
    Datum* datum, bool is_color, const int crop_width, const int crop_height,
    const string& suffix) {

  CHECK_EQ(offsets.size(), bboxes.size());
  string datum_tmp;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

  int id = 0;
  char tmp[30];
  CHECK_GT(bboxes.size(), 0);
  const int num_views = int(bboxes[0].size());
  for (int k = 0; k < num_views; k++) //for views
  {
    for (int i = 0; i < offsets.size(); ++i) //for segments
    {
      int offset = offsets[i];
      const vector<cv::Rect> &bb = bboxes[i];
      for (int file_id = 0; file_id < new_length; ++file_id)
      {
        sprintf(tmp,"%d.",int(file_id+offset));
        string filename_t = filename + "/" + tmp + suffix;

        cv::Mat cv_img_origin = cv::imread(filename_t, cv_read_flag);
        if (!cv_img_origin.data){
          LOG(ERROR) << "Could not load file " << filename_t;
          return false;
        }

        cv::Mat cv_img;
        if (new_height > 0 && new_width > 0 && (new_height != cv_img_origin.rows || new_width != cv_img_origin.cols) ) {
          cv::resize(cv_img_origin, cv_img, cv::Size(new_width, new_height));
        }else{
          cv_img = cv_img_origin;
        }
        cv_img = cv_img(bb[k]);
        cv::resize(cv_img, cv_img, cv::Size(crop_width, crop_height));

        int num_channels = (is_color ? 3 : 1);
        const int hei = cv_img.rows, wid = cv_img.cols;
        if (file_id==0 && i==0)
        {
          datum->set_channels(num_channels*new_length*num_views*offsets.size());
          datum->set_height(hei);
          datum->set_width(wid);
          datum->set_label(label);
          datum->clear_data();
          datum->clear_float_data();
          datum_tmp.resize(hei * wid * datum->channels());
        }

        ReadImage2String(cv_img, datum_tmp, id);
      }
    }
  }
  datum->set_data(datum_tmp);
  return true;
}

bool ReadSegmentFlowToDatum(const string& filename, const int label,
    const vector<int> &offsets, const vector<vector<cv::Rect> > &bboxes,
    const int new_height, const int new_width, const int new_length,
    Datum* datum, const bool is_color, const int crop_width,
    const int crop_height) {

  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
  int num_channels = is_color ? 3 : 1;

  int id = 0;
  char tmp[30];
  CHECK_GT(bboxes.size(), 0);
  const int num_views = int(bboxes[0].size());
  for (int k = 0; k < num_views; k++)
  {
    for (int i = 0; i < offsets.size(); ++i)
    {
      int offset = offsets[i];
      const vector<cv::Rect> &bb = bboxes[i];
      for (int file_id = 0; file_id < new_length; ++file_id)
      {
        sprintf(tmp,"%d_x.png",int(file_id+offset));
        string filename_x = filename + "/" + tmp;
        cv::Mat cv_img_origin_x = cv::imread(filename_x, cv_read_flag);
        sprintf(tmp,"%d_y.png",int(file_id+offset));
        string filename_y = filename + "/" + tmp;
        cv::Mat cv_img_origin_y = cv::imread(filename_y, cv_read_flag);
        if (!cv_img_origin_x.data || !cv_img_origin_y.data){
          LOG(ERROR) << "Could not load file " << filename_x << " or " << filename_y;
          return false;
        }

        cv::Mat cv_img_x, cv_img_y;
        if (new_height > 0 && new_width > 0 && (new_height != cv_img_origin_x.rows ||  new_width != cv_img_origin_x.cols)){
          cv::resize(cv_img_origin_x, cv_img_x, cv::Size(new_width, new_height));
          cv::resize(cv_img_origin_y, cv_img_y, cv::Size(new_width, new_height));
        }else{
          cv_img_x = cv_img_origin_x;
          cv_img_y = cv_img_origin_y;
        }
        cv_img_x = cv_img_x(bb[k]);
        cv_img_y = cv_img_y(bb[k]);
  //      LOG(INFO) << "after roi, size(cv_img_x) = " << cv_img_x.rows << ", " << cv_img_x.cols;
        cv::resize(cv_img_x, cv_img_x, cv::Size(crop_width, crop_height));
        cv::resize(cv_img_y, cv_img_y, cv::Size(crop_width, crop_height));
//        LOG(INFO) << "after resize, size(cv_img_x) = " << cv_img_x.rows << ", " << cv_img_x.cols;

        const int hei = cv_img_x.rows, wid = cv_img_x.cols;
        if (k == 0 && file_id==0 && i==0)
        {
          datum->set_channels(num_channels * 2 * new_length * offsets.size() * num_views);
          datum->set_height(hei);
          datum->set_width(wid);
          datum->set_label(label);
          datum->clear_data();
          datum->clear_float_data();
          datum->mutable_data()->resize(hei * wid * datum->channels());
        }

        string &datum_ref = *(datum->mutable_data());
        ReadImage2String(cv_img_x, datum_ref, id);
        ReadImage2String(cv_img_y, datum_ref, id);
      }
    }
  }
  return true;
}

template <>
void LoadBinFile<float>(const string &path_file, int &pos, vector<float> &data, vector<int> &shape)
{
	FILE  *fid = fopen(path_file.c_str(), "rb");
	CHECK(fid != NULL) << "faied to open file " << path_file;

	shape.resize(4);
	CHECK_EQ(4, fread(&(shape[0]), sizeof(int), 4, fid)) << path_file;
	const int cnt = shape[0] * shape[1] * shape[2] * shape[3];
	if (pos + cnt > int(data.size())) {
		data.resize(data.size() * 2);
	}
	CHECK_EQ(cnt, fread(&(data[pos]), sizeof(float), cnt, fid)) << path_file;

	fclose(fid);
	pos += cnt;
}

template <>
void LoadBinFile<float>(const string &path_file, vector<float> &data, vector<int> &shape)
{
	FILE  *fid = fopen(path_file.c_str(), "rb");
	CHECK(fid != NULL) << "faied to open file " << path_file;

	shape.resize(4);
	CHECK_EQ(4, fread(&(shape[0]), sizeof(int), 4, fid));
	const int cnt = shape[0] * shape[1] * shape[2] * shape[3];
	data.resize(cnt);
	CHECK_EQ(cnt, fread(&(data[0]), sizeof(float), cnt, fid));

	fclose(fid);
}

template <>
void LoadBinFile<double>(const string &path_file, vector<double> &data, vector<int> &shape)
{
	FILE  *fid = fopen(path_file.c_str(), "rb");
	CHECK(fid != NULL) << "faied to open file " << path_file;

	shape.resize(4);
	CHECK_EQ(4, fread(&(shape[0]), sizeof(int), 4, fid));
	const int cnt = shape[0] * shape[1] * shape[2] * shape[3];
	data.resize(cnt);
	CHECK_EQ(cnt, fread(&(data[0]), sizeof(double), cnt, fid));

	fclose(fid);
}

int LoadBinFile(const string &path_file, vector<int> &shape)
{
	FILE  *fid = fopen(path_file.c_str(), "rb");
	CHECK(fid != NULL) << "faied to open file " << path_file;

	shape.resize(4);
	CHECK_EQ(4, fread(&(shape[0]), sizeof(int), 4, fid));
	return shape[0] * shape[1] * shape[2] * shape[3];
}

//shape: num, chn, height, width
template <>
void SaveBinFile<float>(const float *pdata, const vector<int> shape, const string &path_file)
{
	CHECK_EQ(4, shape.size());
	FILE  *fid = fopen(path_file.c_str(), "wb");
	CHECK(fid != NULL) << "faied to open file " << path_file;

	fwrite(&(shape[0]), sizeof(int), 4, fid);
	fwrite(pdata, sizeof(float), shape[0] * shape[1] * shape[2] * shape[3], fid);
	fclose(fid);
}

template<>
void SaveBinFile<float>(const float *pdata, const int num, const int chn, const int height, const int width, const string &path_file)
{
      FILE  *fid = fopen(path_file.c_str(), "wb");
      CHECK(fid != NULL) << "faied to open file " << path_file;

      fwrite(&(num), sizeof(int), 1, fid);
      fwrite(&(chn), sizeof(int), 1, fid);
      fwrite(&(height), sizeof(int), 1, fid);
      fwrite(&(width), sizeof(int), 1, fid);
      fwrite(pdata, sizeof(float), width * height * chn * num, fid);
      fclose(fid);
}

void Linspace(const int s, const int e, const int num, vector<int> &ids)
{
	CHECK(e >= s);
	ids.clear();
	if (1==num)
		ids.push_back((s+e)/2);
	else
	{
		float step = float(e - s) / (num-1);
		for (int i = 0; i < num; i++)
			ids.push_back(int(s + step * i));
	}

}

void Cirspace(const int len, const int num, vector<int> &ids)
{
	CHECK_GT(len, 0);
	const float step = float(len) / num;

	ids.resize(num);
	for (int i = 0; i < num; i++)
		ids[i] = int(step * i + 0.5f) % num;
}

void GenSmpIds(const int len, const int new_length, const int stride_tmp, vector<int> &starts)
{
	const int num = (len - new_length) / stride_tmp + 1;
	starts.resize(num);
	for (int k = 0; k < num; k++)
		starts[k] = stride_tmp * k;
}

void ReadVidList(const string &vidlist, vector<string> &names, vector<int> &lengths, vector<int> &labels)
{
	std::ifstream infile(vidlist.c_str());
	CHECK(infile.is_open()) << "Failed to open " << vidlist << " for reading";
	names.clear();
	lengths.clear();
	labels.clear();

	string filename;
	int label, length;
	while (infile >> filename >> length >> label){
		names.push_back(filename);
		lengths.push_back(length);
		labels.push_back(label);
	}
}

void SaveVidList(const string &vidlist, const vector<string> &names, const vector<int> &lengths, const vector<int> &labels)
{
	std::ofstream outfile(vidlist.c_str());
	CHECK(outfile.is_open()) << "Failed to open " << vidlist << " for reading";

	const size_t num = names.size();
	CHECK_EQ(lengths.size(), num);
	CHECK_EQ(labels.size(), num);

	for (int k = 0; k < num; k++)
		outfile << names[k] << " " << lengths[k] << " " << labels[k] << "\n";
}
void SampleSegments(const int len_fea_arr, const int seq_length, const int stride_seq, const bool is_train, const bool do_rolling, const int num_segments, vector<int> &starts, vector<int> &size_seq, caffe::rng_t* rng)
{
	CHECK_GT(stride_seq, 0);
	CHECK_GT(len_fea_arr, 0);
	starts.resize(num_segments);
	size_seq.resize(num_segments, seq_length);

	const int len_smp = 1 + stride_seq * (seq_length - 1); //length of a single sequence
	if (is_train)
	{
		CHECK_EQ(1, num_segments);
		CHECK(rng != NULL);
		if (do_rolling)
			starts[0] = (*rng)() % len_fea_arr;
		else
			starts[0] = (*rng)() % std::max(stride_seq, len_fea_arr - len_smp + 1);
	} else {

		if (do_rolling)
			Cirspace(len_fea_arr, num_segments, starts);
		else
			Linspace(0, std::max(stride_seq - 1, len_fea_arr - len_smp), num_segments, starts);
	}
	if (!do_rolling) {
		for (int n = 0; n < num_segments; n++)
			size_seq[n] = std::min(seq_length, (len_fea_arr - starts[n] - 1) / stride_seq + 1);
	}
}

void ReadClassNames(const string &path_name_list, vector<string> &names_class)
{
	names_class.clear();

	std::ifstream fin(path_name_list.c_str());
	CHECK(fin.is_open()) << "Failed to open " << path_name_list << " for reading";

	string name;
	while (fin >> name)
		names_class.push_back(name);
}

void SaveTube(const string &tube_path, const vector<vector<int> > &id_seq, const vector<vector<float> > &score_seq, const vector<float> &score_tube)
{
	FILE* pfile = fopen(tube_path.c_str(), "wb");
	CHECK(NULL != pfile);

	const int num_tube = int(id_seq.size());
	CHECK_EQ(1, fwrite(&num_tube, sizeof(int), 1, pfile));

	int len = 0;
	float score = 0.f;
	for (int k = 0; k < num_tube; k++)
	{
		const vector<int> &cur_id_seq = id_seq[k];
		const vector<float> &cur_score_seq = score_seq[k];
		len = int(cur_id_seq.size());
		CHECK_EQ(len, cur_score_seq.size());
		score = score_tube[k];
		CHECK_EQ(1, fwrite(&len, sizeof(int), 1, pfile));
		CHECK_EQ(1, fwrite(&score, sizeof(float), 1, pfile));
		CHECK_EQ(len, fwrite(&(cur_id_seq[0]), sizeof(int), len, pfile));
		CHECK_EQ(len, fwrite(&(cur_score_seq[0]), sizeof(float), len, pfile));
	}
	fclose(pfile);
}

void LoadTube(const string &tube_path, vector<vector<int> > &id_seq, vector<vector<float> > &score_seq, vector<float> &score_tube)
{
	FILE *pfile = fopen(tube_path.c_str(), "rb");
	CHECK(NULL != pfile) << "Failed to read " << tube_path;

	int num_tube = 0;
	CHECK_EQ(1, fread(&num_tube, sizeof(int), 1, pfile));
	id_seq.resize(num_tube);
	score_seq.resize(num_tube);
	score_tube.resize(num_tube);

	int len = 0;
	float score = 0.f;
	for (int k = 0; k < num_tube; k++)
	{
		CHECK_EQ(1, fread(&len, sizeof(int), 1, pfile));
		CHECK_EQ(1, fread(&score, sizeof(float), 1, pfile));
		score_tube[k] = score;

		vector<int> &cur_id_seq = id_seq[k];
		vector<float> &cur_score_seq = score_seq[k];
		cur_id_seq.resize(len);
		cur_score_seq.resize(len);
		CHECK_EQ(len, fread(&(cur_id_seq[0]), sizeof(int), len, pfile));
		CHECK_EQ(len, fread(&(cur_score_seq[0]), sizeof(float), len, pfile));
	}

	fclose(pfile);
}

void DenseSampleBBox(const int width, const int height, const float maxAspectRatio, const float olpstep, const float minAreaRatio, std::vector<cv::Rect> &boxes, std::vector<int> &cumcnt)
{
        // get list of all boxes roughly distributed in grid
        boxes.clear();
        cumcnt.clear();
        cumcnt.push_back(0);

        const float w = 1.f;
        const float h = 1.f;

        //control steps
        const float minSize = sqrt(minAreaRatio);
        const float _scStep = sqrt(1 / olpstep);
        const float _arStep = (1 + olpstep) / (2 * olpstep);
        const float _rcStepRatio = (1 - olpstep) / (1 + olpstep);

        //infer #steps
        const int arRad = int(log(maxAspectRatio) / log(_arStep*_arStep));
        const int scNum = int(ceil(log(std::max(w, h) / minSize) / log(_scStep)));

        for (int s = 0; s<scNum; s++) {
                float a, r, c, bh, bw, kr, kc = -1; float ar, sc;
                for (a = 0; a<2 * arRad + 1; a++) {
                        ar = pow(_arStep, float(a - arRad));
                        sc = minSize*pow(_scStep, float(s));
                        bh = sc / ar; kr = bh * _rcStepRatio;
                        bw = sc * ar; kc = bw * _rcStepRatio;

                        int cnt = 0;
                        for (c = 0.f; c<w - bw + kc; c += kc) {
                                for (r = 0.f; r<h - bh + kr; r += kr) {
                                        const int x = round(width * c);
                                        const int y = round(height * r);
                                        int wid = round( width * std::min(bw, w-c) );
                                        int hei = round( height * std::min(bh, h-r) );
                                        wid = std::min(wid, width - x);
                                        hei = std::min(hei, height - y);
                                        cv::Rect bb(x, y, wid, hei);
                                        boxes.push_back(bb);
                                        cnt++;
                                }
                        }
                        cumcnt.push_back(cumcnt.back() + cnt); //cumulative sum "#bbox_for_type_i = cumcnt[i+1] - cumcnt[i]"
                }
        }
}

int Sample_Sto(const vector<float>& score){
	const float sum = caffe_cpu_asum(score.size(), score.data());
        float rand = 0.;
	caffe_rng_uniform(1, 0.f, 1.f, &rand);
	CHECK_GE(rand, 0.f);
	CHECK_LE(rand, 1.f);
	const float thresh = sum * rand;

	float tmp = 0.f;
	for(int i = 0; i < score.size(); ++i){
		tmp += score[i];
		if (tmp >= thresh) return i;
	}
	CHECK_LT(thresh - tmp, 1e-3);
	return score.size() - 1;
}
}  // namespace caffe

