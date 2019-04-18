#include <iostream>
#include <vector>
#include <string>

#include <boost/regex.hpp>
#include <boost/algorithm/string/regex.hpp>

#include "k_means.hpp"

using namespace std;
using namespace boost;

NdArray<double> VecToNdArray(vector<vector<string>>& lines) {
	NdArray<double> res(lines.size(), lines[0].size());

	for (int i = 0; i < lines.size(); ++i) {
		for (int j = 0; j < lines[i].size(); ++j) {
			res(i, j) = stod(lines[i][j]);
		}
	}

	return res;
}

NdArray<double> LoadDataSet(string filename) {
	ifstream ifs(filename, ios::in);
	if (!ifs.is_open()) 
		throw "File open failed!";
	else {
		string line;
		vector<vector<string>> lines;
		while (getline(ifs, line)) {
			vector<string> items;
			split_regex(items, line, regex("\t"));
			lines.push_back(items);
		}
		return VecToNdArray(lines);

	}
}

int main() {
	NdArray<double> X = LoadDataSet("testset.txt");

	cout << X << endl;
	cluster::KMeans km(4);
	km.fit(X);

	cout << km.centroids << endl;

	/*NdArray<double> X = {
		{1, 1}, {2, 2}, {1, 0}, {3, 2}
	};
	
	NdArray<double> Y(4, 1);
	Y(0, 0) = 0;
	Y(1, 0) = 1;
	Y(2, 0) = 1;
	Y(3, 0) = 2;*/

	std::system("pause");

	return 0;
}