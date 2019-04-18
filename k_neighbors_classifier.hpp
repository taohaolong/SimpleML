#include "NumCpp.hpp"
#include <iostream>
#include <unordered_map>
using namespace std;
using namespace nc;

namespace neighbors {
	class KNeighborsClassifer
	{
	public:
		KNeighborsClassifer(int n_neighbors = 5);
		~KNeighborsClassifer();

		void fit(NdArray<double>& X, NdArray<double>& Y);
		NdArray<double> predict(NdArray<double>& X_test, NdArray<double>& X_train, NdArray<double>& Y);

	private:
		int m_n_neighbors;
		NdArray<double> CalDistance(const NdArray<double>& node, const NdArray<double>& X);
	};

	KNeighborsClassifer::KNeighborsClassifer(int n_neighbors):
		m_n_neighbors(n_neighbors)
	{
	}

	KNeighborsClassifer::~KNeighborsClassifer()
	{
	}

	NdArray<double> KNeighborsClassifer::predict(NdArray<double>& X_test, NdArray<double>& X_train, NdArray<double>& Y_train) {
		Shape s = X_test.shape();
		NdArray<double> res(s.rows, 1);
		for (int i = 0; i < s.rows; ++i) {
			unordered_map<double, int> m;
			double res_class = -1;
			int count = -1;
			NdArray<double> dis = CalDistance(X_test(i, X_test.cSlice()), X_train);
			NdArray<uint32> index = argsort(dis, Axis::ROW);
			for (int j = 0; j < m_n_neighbors; ++j) {
				double key = Y_train(index(j, 0), 0);
				if (m.find(key) == m.end())
					m[key] = 1;
				else
					m[key]++;
				if (count < m[key]) {
					res_class = key;
					count = m[key];
				}
			}

			res(i, 0) = res_class;	
		}

		return res;
	}

	NdArray<double> KNeighborsClassifer::CalDistance(const NdArray<double>& node, const NdArray<double>& X) {
		Shape s = X.shape();
		NdArray<double> dis(s.rows, 1);
		for (int i = 0; i < s.rows; ++i) {
			dis(i, 0) = dot(node - X(i, X.cSlice()), (node - X(i, X.cSlice())).transpose())(0, 0);
		}

		return dis;
	}
}