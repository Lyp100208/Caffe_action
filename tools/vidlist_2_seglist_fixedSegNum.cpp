#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <caffe/util/io.hpp>
using namespace std;
using namespace caffe;

int main(int argc, char ** argv)
{
	if (4 != argc) {
		cout << "Usage: ./vidlist_2_seglist vidlist new_length num_seg" << endl;
		return -1;
	}
	const string &vidlist(argv[1]);
	const int new_length = atoi(argv[2]);
	const int num_seg = atoi(argv[3]);

	vector<string> names;
	vector<int> lengths, labels;
	ReadVidList(vidlist, names, lengths, labels);
	const int num = int(names.size());

	string seglist = vidlist.substr(0, vidlist.find_last_of("."));
	seglist += "_newlength_" + std::to_string(new_length) + "_segNum_" + std::to_string(num_seg) + ".txt";

	ofstream outfile(seglist);
	CHECK(outfile.is_open()) << "Failed to open " << seglist << " for writing";

	vector<int> starts;
	for (int n = 0; n < num; n++) {
		CHECK_GE(lengths[n], new_length);
		Linspace(0, lengths[n]-new_length, num_seg, starts);
		for (int k = 0; k < int(starts.size()); k++)
			outfile << names[n] << " " << labels[n] << " " << starts[k] << " " << lengths[n] << endl;
	}
	return 0;
}
