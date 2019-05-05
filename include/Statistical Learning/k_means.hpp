#ifndef K_MEANS_HPP
#define K_MEANS_HPP

namespace sm {
	namespace cluster {
		class KMeans
		{
		public:
			KMeans(int n_clusters = 8);
			~KMeans();
			void fit(const nc::NdArray<double>& X, const nc::NdArray<double>& Y);
			void fit(const nc::NdArray<double>& X);
			nc::NdArray<double> predict(const nc::NdArray<double>& X);

			int n_clusters_;
			nc::NdArray<double> centroids_;
			nc::NdArray<double> labels_;
		};

		KMeans::KMeans(int n_clusters) : n_clusters_(n_clusters)
		{
		}

		KMeans::~KMeans()
		{
		}

		void KMeans::fit(const nc::NdArray<double>& X, const nc::NdArray<double>& Y) {
			fit(X);
		}

		void KMeans::fit(const nc::NdArray<double>& X) {
			nc::Shape s = X.shape();
			centroids_ = nc::zeros<double>(n_clusters_, s.cols);
			for (int i = 0; i < s.cols; ++i) {
				double min_i = nc::min(X(X.rSlice(), i))(0, 0);
				double max_i = nc::max(X(X.rSlice(), i))(0, 0);
				for (int k = 0; k < n_clusters_; ++k)
					centroids_(k, i) = rand() / double(RAND_MAX) * (max_i - min_i) + min_i;
			}

			labels_ = nc::zeros<double>(s.rows, 2);
			labels_ = -1;
			bool change = true;
			while (change) {
				change = false;

				for (int i = 0; i < s.rows; ++i) {
					double min_dist = INT_MAX;
					int min_index = -1;
					for (int j = 0; j < n_clusters_; ++j) {
						double dist_j = euclidean(X(i, X.cSlice()), centroids_(j, centroids_.cSlice()))(0, 0);
						if (dist_j < min_dist) {
							min_dist = dist_j;
							min_index = j;
						}
					}

					if (int(labels_(i, 0)) != min_index) {
						change = true;
						labels_(i, 0) = min_index;
						labels_(i, 1) = min_dist;
					}
				}

				nc::NdArray<double> count = nc::zeros<double>(n_clusters_, s.cols + 1);
				// 更新质心
				for (int i = 0; i < s.rows; ++i) {
					int k = (int)labels_(i, 0);
					for (int j = 0; j < s.cols; ++j)
						count(k, j) += X(i, j);
					count(k, s.cols) += 1;
				}

				for (int k = 0; k < n_clusters_; ++k)
					for (int j = 0; j < s.cols; ++j)
						centroids_(k, j) = count(k, j) / count(k, s.cols);
			}
		}

		nc::NdArray<double> KMeans::predict(const nc::NdArray<double> & X) {
			nc::Shape s = X.shape();
			nc::NdArray<double> Y_predict(s.rows, 1);
			for (int i = 0; i < s.rows; ++i) {
				double min_dist = INT_MAX;
				int min_index = -1;
				for (int k = 0; k < n_clusters_; ++k) {
					double dist = euclidean(X(i, X.cSlice()), centroids_(k, centroids_.cSlice()))(0, 0);
					if (dist < min_dist) {
						min_dist = dist;
						min_index = k;
					}
				}
				Y_predict(i, 0) = min_index;
			}

			return Y_predict;
		}
	}
}
#endif