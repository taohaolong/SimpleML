#include <iostream>
#include <boost/regex.hpp>
#include <boost/algorithm/string/regex.hpp>

#include "NumCpp.hpp"

using namespace std;
//#include "include/SimpleML.hpp"
//
//using namespace std;
//
//nc::NdArray<double> VecToNdArray(vector<vector<string>>& lines) {
//	nc::NdArray<double> res(lines.size(), lines[0].size());
//
//	for (int i = 0; i < lines.size(); ++i) {
//		for (int j = 0; j < lines[i].size(); ++j) {
//			res(i, j) = stod(lines[i][j]);
//		}
//	}
//
//	return res;
//}
//
//nc::NdArray<double> LoadDataSet(string filename) {
//	ifstream ifs(filename, ios::in);
//	if (!ifs.is_open()) 
//		throw "File open failed!";
//	else {
//		string line;
//		vector<vector<string>> lines;
//		while (getline(ifs, line)) {
//			vector<string> items;
//			boost::split_regex(items, line, boost::regex("\t"));
//			lines.push_back(items);
//		}
//		return VecToNdArray(lines);
//
//	}
//}

int main() { 
	nc::NdArray<double> X = { {1, 2}, {3, 4}, {5, 6} };
	nc::NdArray<double> Y = nc::ones<double>(1, 2);
	nc::NdArray<double> dis = nc::sum((X - Y) * (X - Y), nc::Axis::COL);
	dis.reshape(dis.shape().cols, dis.shape().rows);
	cout << dis << endl;

	std::system("pause");

	return 0;
}