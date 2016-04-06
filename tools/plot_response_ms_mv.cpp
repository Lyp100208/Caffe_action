#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include "glog/logging.h"
#include "caffe/util/io.hpp"
using namespace caffe;
#include "caffe/util/file_helper.hpp"
using namespace std;

float Analysis(const vector<float> &score, const int num_class, const int num_views, const int num_seg, const int label_gt, vector<vector<float> > &prob_gt);
void AverageCurve(const vector<vector<float> > &probs, vector<float> &meanp);
void DrawResponse(const vector<float> &scores, const vector<int> &starts, const vector<int> &length, const int num_frame, cv::Mat &curve, const cv::Scalar &color);
void MarkCurve(const cv::Mat &curve0, const int num_frame, const int pos, cv::Mat &curve);

void Img2Flow(const cv::Mat &img, const float range, cv::Mat &flow);
void VisualizeOptFlow(cv::Mat opt_x, cv::Mat opt_y, cv::Mat &visImg, double globalMaxMag);
void GetFlowVisImg(const string &root_flow, const int id_frame, const string &name_vid, const int wid_img, const int hei_img, cv::Mat &flowImg);

int main(int argc, char *argv[])
{
	if (13 != argc) {
		cout << "Usage: ./plot_response_ms_mv vidlist classlist root_img root_flow root_score root_vis num_seg num_views new_length seq_length stride_seq do_rolling" << endl;
		return -1;
	}
	const string vidlist(argv[1]);
	const string classlist(argv[2]);
	const string root_img(argv[3]); //rgb image folder
	const string root_flow(argv[4]);
	const string root_score(argv[5]); //score files
	const string root_vis(argv[6]); //folder for visualization result
	const int num_seg = atoi(argv[7]);
	const int num_views = atoi(argv[8]);
	const int new_length = atoi(argv[9]); //length of a segment
	const int seq_length = atoi(argv[10]); //seqence length in segment feature array
	const int stride_seq = atoi(argv[11]); //sequece stride in segment feature array
	const int do_rolling = atoi(argv[12]);

	//fixed paras
	const bool is_train = false;
	const int width = 320;
	const int height = 240;

	char path[160];
	sprintf(path, "%s_%dsegs_%dviews_seqlen%d_strideseq%d_rolling%d", root_vis.c_str(), num_seg, num_views, seq_length, stride_seq, do_rolling);
	CheckOutputDir(root_vis);

	//read list
	vector<string> names_vid;
	vector<int> lens_vid, labels_gt;
	ReadVidList(vidlist, names_vid, lens_vid, labels_gt);
	const int num_vid = int(names_vid.size());
	LOG(INFO) << "Get " << num_vid  << " videos from " << vidlist;

	//label_id --> class_name
	vector<string> names_class;
	ReadClassNames(classlist, names_class);
	const int num_class = int(names_class.size());
	LOG(INFO) << "Get " << names_class.size() << " classes from " << classlist;
	
	for (int i = 0; i < num_vid; ++i)
	{
		LOG(INFO) << i << "/" << num_vid << ": " << names_vid[i];
		const string &name_vid = names_vid[i];
		const int label_gt = labels_gt[i];
		const int len_vid = lens_vid[i];
		const int len_fea_array = len_vid - new_length + 1; //assume stride = 1 in segment feature computing

		//read score file
		char path[260];
		sprintf(path, "%s/%s_%dsegs_%dviews.bin", root_score.c_str(), name_vid.c_str(), num_seg, num_views);
		vector<int> shape;
		vector<float> scores;
		LoadBinFile(path, scores, shape);
		CHECK_EQ(shape[0], num_views * num_seg);
		CHECK_EQ(shape[1], num_class);
		CHECK_EQ(shape[2], 1);
		CHECK_EQ(shape[3], 1);

		//get response curve
		vector<vector<float> > prob_gt;
		const float mean_score_vid = Analysis(scores, num_class, num_views, num_seg, label_gt, prob_gt);
		vector<float> meanprob;
		AverageCurve(prob_gt, meanprob);

		//get frame-range for each segment
		vector<int> starts;
		vector<int> size_seq;
		SampleSegments(len_fea_array, seq_length, stride_seq, is_train, do_rolling, num_seg, starts, size_seq); //length in feature array
		vector<int> length(num_seg);
		for (int k = 0; k < num_seg; k++) {
			length[k] = new_length + stride_seq * (size_seq[k] - 1);	//length in original frames
		}

		//draw the whole response curve
		cv::Mat curve0(cv::Size(width * 2, 150), CV_8UC3, cv::Scalar(255, 255, 255));
		DrawResponse(meanprob, starts, length, len_vid, curve0, cv::Scalar(128, 128, 128));

		//initialize video writer
		sprintf(path, "%s/%.2f_%s.avi", root_vis.c_str(), mean_score_vid, name_vid.c_str());
		if (ExistFile(path))
			continue;
		cv::Size vidsize(width * 2, height + curve0.rows);
		cv::VideoWriter writer(std::string(path), CV_FOURCC('D', 'I', 'V', 'X'), 15, vidsize);
		CHECK(writer.isOpened()) << "Failed to open video " << path;

		// do visualization frame-by-frame
		for (int n = 0; n < len_vid; ++n)
		{
			// read rgb image
			char path[260];
			sprintf(path, "%s/%s/%d.jpg", root_img.c_str(), name_vid.c_str(), n);
			cv::Mat rgb = cv::imread(path);
			CHECK(!rgb.empty()) << "Failed to open image " << path << endl;
			cv::resize(rgb, rgb, cv::Size(width, height));

			// get flow visualization images
			cv::Mat flowImg;
			GetFlowVisImg(root_flow, n, name_vid, width, height, flowImg);

			// mark curve
			cv::Mat curve;
			MarkCurve(curve0, len_vid, n, curve);

			//concatenate images
			cv::hconcat(rgb, flowImg, rgb);
			cv::vconcat(rgb, curve, rgb);
			CHECK_EQ(rgb.cols, vidsize.width);
			CHECK_EQ(rgb.rows, vidsize.height);
			writer << rgb;
		}
		writer.release();
	}
	return 0;
}

void Img2Flow(const cv::Mat &img, const float range, cv::Mat &flow)
{
	if (img.type() != CV_8UC1)
	{
		printf("image should be type of CV_8UC1\n");
		exit(-1);
	}

	const int height = img.rows, width = img.cols;
	flow.create(height, width, CV_32FC1);
	for (int y = 0; y < height; y++)
	{
		float *prow_flow = flow.ptr<float>(y);
		const uchar *prow_img = img.ptr<uchar>(y);
		for (int x = 0; x < width; x++)
			prow_flow[x] = float(prow_img[x]) / 255.f * 2.f * range - range;
	}
}

void VisualizeOptFlow(cv::Mat opt_x, cv::Mat opt_y, cv::Mat &visImg, double globalMaxMag)
{
	assert(globalMaxMag > 0);
	//calculate angle and magnitude
	cv::Mat magnitude, angle;
	cv::cartToPolar(opt_x, opt_y, magnitude, angle, true);//ture to return angle in degree

	magnitude.convertTo(magnitude, -1, 1.0 / globalMaxMag);

	//build hsv image
	cv::Mat _hsv[3], hsv;
	_hsv[0] = angle;
	_hsv[1] = magnitude;
	_hsv[2] = cv::Mat::ones(angle.size(), CV_MAKETYPE(angle.depth(), 1));
	cv::merge(_hsv, 3, hsv);

	//convert to BGR and show
	cv::cvtColor(hsv, visImg, cv::COLOR_HSV2BGR);
	visImg.convertTo(visImg, CV_8UC3, 255);
}

//num_seg * num_view * #class == score.size()
float Analysis(const vector<float> &score, const int num_class, const int num_views, const int num_seg, const int label_gt, vector<vector<float> > &prob_gt)
{
	prob_gt.resize(num_views);
	for (int v = 0; v < num_views; v++)
		prob_gt[v].resize(num_seg);

	CHECK_EQ(score.size(), num_class * num_views * num_seg);

	const float *pscore = &(score[0]);
	float meanscore = 0.f;
	for (int s = 0; s < num_seg; s++) {
		for (int v = 0; v < num_views; v++) {
			const float scor = pscore[label_gt];
			CHECK_GE(scor, 0.f);
			CHECK_LE(scor, 1.f);
			prob_gt[v][s] = scor;
			meanscore += scor;
			pscore += num_class;
		}
	}
	meanscore /= (num_views * num_seg);
	return meanscore;
}

void AverageCurve(const vector<vector<float> > &probs, vector<float> &meanp)
{
	const int num = int(probs.size());
	CHECK_GT(probs.size(), 0);
	const int len = int(probs[0].size());

	meanp.resize(len, 0.f);
	for (int k = 0; k < num; k++) {
		const vector<float> &prob = probs[k];
		CHECK_EQ(len, prob.size());
		for (int i = 0; i < len; i++)
			meanp[i] += prob[i] / num;
	}
}
void DrawResponse(const vector<float> &scores, const vector<int> &starts, const vector<int> &length, const int num_frame, cv::Mat &curve, const cv::Scalar &color)
{
	const int num_seg = int(scores.size());
	CHECK_EQ(num_seg, starts.size());
	CHECK_EQ(num_seg, length.size());

	const int hei = curve.rows, wid = curve.cols;
	const float ratiox = float(wid-1) / (num_frame-1);
	const float ratioy = float(hei-1) / 1.f; //maximum score is 1.f

	for (int n = 0; n < num_seg; n++) {
		const int x0 = round(starts[n] * ratiox);
		const int x1 = round((starts[n] + length[n] - 1) * ratiox);
		const int y = round((1.f - scores[n]) * ratioy);
		cv::line(curve, cv::Point2f(x0, y), cv::Point2f(x1, y), color);
	}
}

void MarkCurve(const cv::Mat &curve0, const int num_frame, const int pos, cv::Mat &curve)
{
	const int width = curve0.cols, height = curve0.rows;
	curve0.copyTo(curve);

	const float ratiox = float(width-1) / (num_frame-1);
	const int x = int (pos * ratiox);
	cv::line(curve, cv::Point(x, 0), cv::Point(x, height-1), cv::Scalar(0));
}


void GetFlowVisImg(const string &root_flow, const int id_frame, const string &name_vid, const int wid_img, const int hei_img, cv::Mat &flowImg)
{
	char path[260];
	sprintf(path, "%s/%s/%d_vis.jpg", root_flow.c_str(), name_vid.c_str(), id_frame);
	if (ExistFile(path)) {
		flowImg = cv::imread(path);
	} else {
		cv::Mat flow_x;
		sprintf(path, "%s/%s/%d_x.png", root_flow.c_str(), name_vid.c_str(), id_frame);
		cv::Mat tmp = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
		CHECK(!tmp.empty()) << "Failed to open image " << path << endl;
		Img2Flow(tmp, 15.f, flow_x);

		cv::Mat flow_y;
		sprintf(path, "%s/%s/%d_y.png", root_flow.c_str(), name_vid.c_str(), id_frame);
		tmp = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
		CHECK(!tmp.empty()) << "Failed to open image " << path << endl;
		Img2Flow(tmp, 15.f, flow_y);

		VisualizeOptFlow(flow_x, flow_y, flowImg, 10.f);		
	}

	if (hei_img != flowImg.rows || wid_img != flowImg.cols)
	cv::resize(flowImg, flowImg, cv::Size(wid_img, hei_img));
}

