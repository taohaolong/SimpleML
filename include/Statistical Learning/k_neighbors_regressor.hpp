#ifndef K_NEIGHBORS_REGRESSOR_HPP
#define K_NEIGHBORS_REGRESSOR_HPP

namespace sm {
	namespace neighbors {
		class KNeighborsRegressor
		{
		public:
			KNeighborsRegressor(int n_neighbors = 5);
			~KNeighborsRegressor();

			void fit(const nc::NdArray<double>& X, const nc::NdArray<double>& Y);
			nc::NdArray<double> predict(const nc::NdArray<double>& X_test, const nc::NdArray<double>& X_train, const nc::NdArray<double>& Y);

		private:
			int n_neighbors_;
		};

		KNeighborsRegressor::KNeighborsRegressor(int n_neighbors) :
			n_neighbors_(n_neighbors)
		{
		}

		KNeighborsRegressor::~KNeighborsRegressor()
		{
		}

		nc::NdArray<double> KNeighborsRegressor::predict(const nc::NdArray<double>& X_test, const nc::NdArray<double>& X_train, const nc::NdArray<double>& Y_train) {
			nc::Shape s = X_test.shape();
			nc::NdArray<double> res(s.rows, 1);
			for (int i = 0; i < s.rows; ++i) {
				double res_value = 0;
				nc::NdArray<double> dis = euclidean(X_test(i, X_test.cSlice()), X_train);
				nc::NdArray<nc::uint32> index = nc::argsort(dis, nc::Axis::ROW);
				for (int j = 0; j < n_neighbors_; ++j) {
					res_value += Y_train(index(j, 0), 0);

				}

				res(i, 0) = res_value / n_neighbors_;
			}

			return res;
		}
	}
}

#endif