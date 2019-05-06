#define _CRT_SECURE_NO_WARNINGS
#include <iostream>

#include "NumCpp.hpp"
#include "include/SimpleML.hpp"

using namespace std;

int main() { 
	nc::NdArray<double> X = sm::load_dataset<double>("data/kmeans_test.txt", "\t");
	
	sm::cluster::KMeans km(4);

	km.fit(X);

	cout << km.centroids_ << endl;

	std::system("pause");

	return 0;
}