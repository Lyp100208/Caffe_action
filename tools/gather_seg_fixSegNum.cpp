#include <iostream>
#include <glog/logging.h>
#include <string>
#include <vector>
using namespace std;
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include "caffe/util/io.hpp"
using namespace caffe;


int main(int argc, char **argv)
{
	if (7 != argc) {
		cout << "Usage: ./gather_view_data folder_sep folder_gather vidlist new_length num_seg num_views" << endl;
		return -1;
	}

	const string folder_sep(argv[1]);
	const string folder_gather(argv[2]);
	const string list_file(argv[3]);
	const int new_length = atoi(argv[4]);
	const int num_seg = atoi(argv[5]);
	const int num_views = atoi(argv[6]);
	const string tmp = "mkdir -p " + folder_gather;
	CHECK(-1 != system(tmp.c_str()));

	std::ifstream infile(list_file.c_str());
	if (!infile.is_open()) {
		cout << "Failed to open " << list_file << endl;
		return -1;
	}

	char path[260];
	vector<float> data(20000000);
	vector<int> starts;
	string filename;
	int length, label;
	while (infile >> filename >> length >> label){
		Linspace(0, length - new_length, num_seg, starts);
		
		const int num_file = int(starts.size());
		CHECK_GT(num_file, 0) << "Get no .bin files for video: " << filename;

		int pos = 0;
		vector<int> s, s0;
		for (int n = 0; n < num_file; n++) {
			sprintf(path, "%s/%s_%d_%dviews.bin", folder_sep.c_str(), filename.c_str(), starts[n], num_views);
			LoadBinFile(path, pos, data, s);
			if (0 == n) {
				s0 = s;
			} else {
				CHECK(s == s0);
			}
		}
		CHECK_EQ(num_views, s[0]);
		CHECK_EQ(1, s[2]);
		CHECK_EQ(1, s[3]);

		s[0] *= num_file;
		sprintf(path, "%s/%s_%dsegs_%dviews.bin", folder_gather.c_str(), filename.c_str(), num_file, num_views);
		SaveBinFile(&(data[0]), s, path);
	}
	return 0;
}
