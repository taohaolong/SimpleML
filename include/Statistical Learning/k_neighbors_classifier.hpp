#ifndef K_NEIGHBORS_CLASSIFIER_HPP
#define K_NEIGHBORS_CLASSIFIER_HPP

namespace sm {
	namespace neighbors {
		class KNeighborsClassifer
		{
		public:
			KNeighborsClassifer(int n_neighbors = 5);
			~KNeighborsClassifer();

			void fit(const nc::NdArray<double>& X, const nc::NdArray<double>& Y);
			nc::NdArray<double> predict(const nc::NdArray<double>& X_test, const nc::NdArray<double>& X_train, const nc::NdArray<double>& Y);

		private:
			int n_neighbors_;
		};

		KNeighborsClassifer::KNeighborsClassifer(int n_neighbors) :
			n_neighbors_(n_neighbors)
		{
		}

		KNeighborsClassifer::~KNeighborsClassifer()
		{
		}

		nc::NdArray<double> KNeighborsClassifer::predict(const nc::NdArray<double>& X_test, const nc::NdArray<double>& X_train, const nc::NdArray<double>& Y_train) {
			nc::Shape s = X_test.shape();
			nc::NdArray<double> res(s.rows, 1);
			for (int i = 0; i < s.rows; ++i) {
				std::unordered_map<double, int> m;
				double res_class = -1;
				int count = -1;
				nc::NdArray<double> dis = euclidean(X_test(i, X_test.cSlice()), X_train);
				nc::NdArray<nc::uint32> index = nc::argsort(dis, nc::Axis::ROW);
				for (int j = 0; j < n_neighbors_; ++j) {
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
	}
}

#endif