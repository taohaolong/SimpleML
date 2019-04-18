#include "NumCpp.hpp"
using namespace nc;

namespace neighbors {
	class KNeighborsRegressor
	{
	public:
		KNeighborsRegressor(int n_neighbors = 5);
		~KNeighborsRegressor();

		void fit(NdArray<double>& X, NdArray<double>& Y);
		NdArray<double> predict(NdArray<double>& X_test, NdArray<double>& X_train, NdArray<double>& Y);

	private:
		int m_n_neighbors;
		NdArray<double> CalDistance(const NdArray<double>& node, const NdArray<double>& X);
	};

	KNeighborsRegressor::KNeighborsRegressor(int n_neighbors) :
		m_n_neighbors(n_neighbors)
	{
	}

	KNeighborsRegressor::~KNeighborsRegressor()
	{
	}

	NdArray<double> KNeighborsRegressor::predict(NdArray<double>& X_test, NdArray<double>& X_train, NdArray<double>& Y_train) {
		Shape s = X_test.shape();
		NdArray<double> res(s.rows, 1);
		for (int i = 0; i < s.rows; ++i) {
			double res_value = 0;
			NdArray<double> dis = CalDistance(X_test(i, X_test.cSlice()), X_train);
			NdArray<uint32> index = argsort(dis, Axis::ROW);
			for (int j = 0; j < m_n_neighbors; ++j) {
				res_value += Y_train(index(j, 0), 0);
				
			}

			res(i, 0) = res_value / m_n_neighbors;
		}

		return res;
	}

	NdArray<double> KNeighborsRegressor::CalDistance(const NdArray<double>& node, const NdArray<double>& X) {
		Shape s = X.shape();
		NdArray<double> dis(s.rows, 1);
		for (int i = 0; i < s.rows; ++i) {
			dis(i, 0) = dot(node - X(i, X.cSlice()), (node - X(i, X.cSlice())).transpose())(0, 0);
		}

		return dis;
	}
}