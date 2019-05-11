#ifndef GAUSSIAN_MIXTURE_HPP
#define GAUSSIAN_MIXTURE_HPP

namespace sm {
	namespace mixture {

		class GaussianMixture
		{
		public:
			nc::NdArray<double> weights_;
			nc::NdArray<double> means_;
			nc::DataCube<double> covariances_;
			nc::NdArray<double> lower_bound_;

			GaussianMixture(int n_components = 1, double tol = 1e-3, int max_iter = 100);
			~GaussianMixture();

			void fit(const nc::NdArray<double>& X, const nc::NdArray<double>& Y);
			void fit(const nc::NdArray<double>& X);
			nc::NdArray<int> predict(const nc::NdArray<double>& X);

		private:
			int n_components_;
			double tol_;
			int max_iter_;

			void init_params(const nc::NdArray<double>& X);
		};

		GaussianMixture::GaussianMixture(int n_components, double tol, int max_iter):
			n_components_(n_components),
			tol_(tol),
			max_iter_(max_iter)
		{

		}

		GaussianMixture::~GaussianMixture()
		{

		}

		void GaussianMixture::init_params(const nc::NdArray<double>& X)
		{
			nc::Shape s = X.shape();
			nc::NdArray<double> X_min = X.min(nc::Axis::ROW);
			nc::NdArray<double> X_max = X.max(nc::Axis::ROW);

			// 初始化均值和协方差矩阵
			means_ = nc::NdArray<double>(n_components_, s.cols);
			for (int col = 0; col < s.cols; ++col)
			{
				means_.assignCol(col, nc::Random<double>::randFloat(nc::Shape(n_components_, 1), X_min[col], X_max[col]));
			}
	
			for (int i = 0; i < n_components_; ++i)
			{
				covariances_.push_back(nc::identity<double>(s.cols));
			}

			// 初始化分布权重
			weights_ = nc::Random<double>::rand(nc::Shape(n_components_, 1));
			weights_ = weights_ / nc::sum(weights_);
		}

		void GaussianMixture::fit(const nc::NdArray<double>& X, const nc::NdArray<double>& Y)
		{
			fit(X);
		}

		void GaussianMixture::fit(const nc::NdArray<double>& X)
		{
			nc::Shape s = X.shape();
			init_params(X);

			for (int iter = 0; iter < max_iter_; ++iter)
			{
				// E step
				// 1. 计算概率值
				nc::NdArray<double> tau(n_components_, s.rows);
				for (int i = 0; i < n_components_; ++i) 
				{
					tau.assignRow(i, multivariate_normal(X, means_(i, means_.cSlice()), covariances_[i]));
				}

				tau = tau * weights_;

				// 2. 概率值均一化（即公式中的w）
				tau = tau / tau.sum<double>(nc::Axis::ROW);

				// M step
				nc::NdArray<double> tau_sum = tau.sum<double>(nc::Axis::COL);

				// 1. 计算更新后的均值
				nc::NdArray<double> new_means = nc::dot(tau, X) / tau_sum;

				if (abs(nc::max(means_ - new_means)(0, 0)) < tol_)
					break;

				means_ = nc::dot(tau, X) / tau_sum;

				// 2. 计算更新后的方差
				for (int i = 0; i < n_components_; ++i) {
					covariances_[i] = nc::dot( ( X - means_(i, means_.cSlice()) ).transpose() * tau(i, tau.cSlice()), X - means_(i, means_.cSlice()) ) / tau_sum(i, 0);
				}
				// 3. 计算更新后的分布权重
				weights_ = tau_sum / s.rows;
			}
		}

		nc::NdArray<int> GaussianMixture::predict(const nc::NdArray<double>& X)
		{
			nc::Shape s = X.shape();
			nc::NdArray<int> Y_predict(s.rows, 1);

			for (int row = 0; row < s.rows; ++row)
			{
				nc::NdArray<double> dist = euclidean(X(row, X.cSlice()), means_);
				Y_predict(row, 0) = nc::argmin(dist, nc::Axis::ROW)(0, 0);
			}

			return Y_predict;
		}

	}
}
#endif