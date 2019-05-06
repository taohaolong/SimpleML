#ifndef K_MEANS_HPP
#define K_MEANS_HPP

namespace sm {
	namespace cluster {
		class KMeans
		{
		public:
			KMeans(int n_clusters = 8, int max_iter = 100);
			~KMeans();
			void fit(const nc::NdArray<double>& X, const nc::NdArray<int>& Y);
			void fit(const nc::NdArray<double>& X);
			nc::NdArray<int> predict(const nc::NdArray<double>& X);

			int n_clusters_;
			int max_iter_;
			nc::NdArray<double> centroids_;
			nc::NdArray<int> labels_;

		private:
			void init_centroids(const nc::NdArray<double>& X);
		};

		KMeans::KMeans(int n_clusters, int max_iter) : 
			n_clusters_(n_clusters),
			max_iter_(max_iter)
		{
		}

		KMeans::~KMeans()
		{
		}

		void KMeans::init_centroids(const nc::NdArray<double>& X) {
			nc::Shape s = X.shape();
			centroids_ = nc::NdArray<double>(n_clusters_, s.cols);

			nc::NdArray<double> X_min = nc::min(X, nc::Axis::ROW);
			nc::NdArray<double> X_max = nc::max(X, nc::Axis::ROW);
			for (int col = 0; col < s.cols; ++col) {
				centroids_.assignCol(col, nc::Random<double>::randFloat(nc::Shape(n_clusters_, 1), X_min(0, col), X_max(0, col)));
			}
		}

		void KMeans::fit(const nc::NdArray<double>& X, const nc::NdArray<int>& Y) {
			fit(X);
		}

		void KMeans::fit(const nc::NdArray<double>& X) {
			nc::Shape s = X.shape();

			init_centroids(X);
			labels_ = nc::nans<int>(s.rows, 1);

			nc::NdArray<double> dist = nc::zeros<double>(s.rows, n_clusters_);

			for (int iter = 0; iter < max_iter_; ++iter) {
				bool change = false;

				//将每个样本点分配到最近的簇中
				for (int row = 0; row < s.rows; ++row) {
					nc::NdArray<double> cur_dist = euclidean(X(row, X.cSlice()), centroids_);
					cur_dist.reshape(1, n_clusters_);
					dist.assignRow(row, cur_dist);
					int next_index = nc::argmin(cur_dist, nc::Axis::COL)(0, 0);
					if (labels_[row] != next_index) {
						change = true;
						labels_[row] = next_index;
					}
				}

				//如果质心不再改变，则跳出循环
				if (!change)
					break;
				
				//更新质心
				for (int k = 0; k < n_clusters_; ++k) {
					nc::NdArray<double> cluster_k = X[labels_ == k];
					centroids_.assignRow(k, nc::sum(cluster_k, nc::Axis::COL) / cluster_k.shape().cols);
				}
			}
		}

		nc::NdArray<int> KMeans::predict(const nc::NdArray<double> & X) {
			nc::Shape s = X.shape();
			nc::NdArray<int> Y_predict(s.rows, 1);
			for (int row = 0; row < s.rows; ++row) {
				nc::NdArray<double> cur_dist = euclidean(X(row, X.cSlice()), centroids_);
				Y_predict[row] = nc::argmin(cur_dist, nc::Axis::ROW)(0, 0);
			}

			return Y_predict;
		}
	}
}
#endif