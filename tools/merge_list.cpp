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
		cout << "Usage: ./merge_list list_dst list_src1 list_src2" << endl;
		return -1;
	}
	const string &list_dst(argv[1]);
	const string &list_src1(argv[2]);
	const string &list_src2(argv[3]);

	vector<string> names1;
	vector<int> lengths1, labels1;
	ReadVidList(list_src1, names1, lengths1, labels1);
	const int num1 = int(names1.size());
	LOG(INFO) << "Get " << num1 << " videos in " << list_src1;

	vector<string> names2;
	vector<int> lengths2, labels2;
	ReadVidList(list_src2, names2, lengths2, labels2);
	const int num2 = int(names2.size());
	LOG(INFO) << "Get " << num2 << " videos in " << list_src2;

	names1.insert(names1.end(), names2.begin(), names2.end());
	lengths1.insert(lengths1.end(), lengths2.begin(), lengths2.end());
	labels1.insert(labels1.end(), labels2.begin(), labels2.end());

	SaveVidList(list_dst, names1, lengths1, labels1);

	return 0;
}
