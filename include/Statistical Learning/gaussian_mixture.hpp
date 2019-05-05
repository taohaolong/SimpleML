#ifndef GAUSSIAN_MIXTURE_HPP
#define GAUSSIAN_MIXTURE_HPP

namespace sm {
	namespace mixture {

		class GaussianMixture
		{
		public:
			nc::NdArray<double> weights_;
			nc::NdArray<double> means_;
			std::vector<nc::NdArray<double>> covariances_;
			nc::NdArray<double> lower_bound_;

			GaussianMixture(int n_components = 1, double tol = 1e-3, int max_iter = 100);
			~GaussianMixture();

			void fit(const nc::NdArray<double>& X);
			nc::NdArray<double> predict(const nc::NdArray<double>& X);

		private:
			int n_components_;
			double tol_;
			int max_iter_;
		};

	}
}
#endif