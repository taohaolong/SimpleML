#define _CRT_SECURE_NO_WARNINGS
#include <iostream>

#include "NumCpp.hpp"
#include "include/SimpleML.hpp"

using namespace std;

int main() { 
	nc::NdArray<double> data = sm::load_dataset<double>("data/dtree_test.txt");
	nc::NdArray<double> X = data(data.rSlice(), nc::Slice(0, 2));
	nc::NdArray<int> Y = data(data.rSlice(), data.cSlice(2)).astype<int>();
	
	sm::tree::DecisionTreeClassifier dt;
	dt.fit(X, Y);
	sm::tree::print_tree(dt.root_);

	std::system("pause");

	return 0;
}