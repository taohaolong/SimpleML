#include "NumCpp.hpp"
using namespace nc;

namespace cluster {
	class KMeans
	{
	public:
		KMeans(int n_clusters = 8);
		~KMeans();
		void fit(NdArray<double>& X, NdArray<double>& Y);
		void fit(NdArray<double>& X);
		NdArray<double> predict(NdArray<double>& X);

		int n_clusters;
		NdArray<double> centroids;
		NdArray<double> labels;
	private:
		double CalDist(const NdArray<double>& x, const NdArray<double>& y);
	};

	KMeans::KMeans(int n_clusters): n_clusters(n_clusters)
	{
	}

	KMeans::~KMeans()
	{
	}

	double KMeans::CalDist(const NdArray<double>& x, const NdArray<double>& y) {
		return sum(power(x - y, 2))(0, 0);
	}

	void KMeans::fit(NdArray<double>& X, NdArray<double>& Y) {
		fit(X);
	}

	void KMeans::fit(NdArray<double>& X) {
		Shape s = X.shape();
		centroids = zeros<double>(n_clusters, s.cols);
		for (int i = 0; i < s.cols; ++i) {
			double min_i = min(X(X.rSlice(), i))(0, 0);
			double max_i = max(X(X.rSlice(), i))(0, 0);
			for(int k = 0; k < n_clusters; ++k)
				centroids(k, i) = rand() / double(RAND_MAX) * (max_i - min_i) + min_i;
		}

		labels = zeros<double>(s.rows, 2);
		labels = -1;
		bool change = true;
		while (change) {
			change = false;

			for (int i = 0; i < s.rows; ++i) {
				double min_dist = INT_MAX;
				int min_index = -1;
				for (int j = 0; j < n_clusters; ++j) {
					double dist_j = CalDist(X(i, X.cSlice()), centroids(j, centroids.cSlice()));
					if (dist_j < min_dist) {
						min_dist = dist_j;
						min_index = j;
					}
				}

				if (int(labels(i, 0)) != min_index) {
					change = true;
					labels(i, 0) = min_index;
					labels(i, 1) = min_dist;
				}
			}

			NdArray<double> count = zeros<double>(n_clusters, s.cols+1);
			// 更新质心
			for (int i = 0; i < s.rows; ++i) {
				int k = (int)labels(i, 0);
				for (int j = 0; j < s.cols; ++j)
					count(k, j) += X(i, j);
				count(k, s.cols) += 1;
			}

			for (int k = 0; k < n_clusters; ++k)
				for (int j = 0; j < s.cols; ++j)
					centroids(k, j) = count(k, j) / count(k, s.cols);
		}
	}

	NdArray<double> KMeans::predict(NdArray<double>& X) {
		Shape s = X.shape();
		NdArray<double> Y_predict(s.rows, 1);
		for (int i = 0; i < s.rows; ++i) {
			double min_dist = INT_MAX;
			int min_index = -1;
			for (int k = 0; k < n_clusters; ++k) {
				double dist = CalDist(X(i, X.cSlice()), centroids(k, centroids.cSlice()));
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