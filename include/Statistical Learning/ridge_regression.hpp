#ifndef RIDGE_REGRESSION_HPP
#define RIDGE_REGRESSION_HPP

namespace sm {
	namespace linear_model {
		class RidgeRegression
		{
		public:
			nc::NdArray<double> coef_;
			double intercept_;

			RidgeRegression(bool fit_intercept = true, double lambda = 0.2);
			~RidgeRegression();

			void fit(nc::NdArray<double>& X, nc::NdArray<double>& y);
			nc::NdArray<double> predict(nc::NdArray<double>& X);
			double score(nc::NdArray<double>& X, nc::NdArray<double>& y);

		private:
			bool fit_intercept_;
			double m_lambda;
		};

		RidgeRegression::RidgeRegression(bool fit_intercept, double lambda) :
			fit_intercept_(fit_intercept),
			m_lambda(lambda)
		{	}

		RidgeRegression::~RidgeRegression()
		{	}

		void RidgeRegression::fit(nc::NdArray<double>& X, nc::NdArray<double>& y) {
			//coef_ = (X^T * X) ^ { -1 } * X^T * Y 
			nc::NdArray<double> X_temp;

			if (fit_intercept_)
				X_temp = nc::hstack({ nc::ones<double>(X.shape().rows, 1), X });
			else
				X_temp = X;

			nc::NdArray<double> denom = nc::dot(X_temp.transpose(), X_temp) + nc::eye<double>(X_temp.shape().cols);

			nc::NdArray<double> theta = nc::linalg::multi_dot({ nc::linalg::inv(denom), X_temp.transpose(), y });

			if (fit_intercept_) {
				intercept_ = theta(0, 0);
				coef_ = theta(theta.rSlice(1), 0);
			}
			else {
				intercept_ = 0;
				coef_ = theta;
			}

		}

		nc::NdArray<double> RidgeRegression::predict(nc::NdArray<double>& X) {
			return nc::dot(X, coef_) + intercept_;
		}

		double RidgeRegression::score(nc::NdArray<double>& X, nc::NdArray<double>& y) {
			return nc::mean(predict(X) == y)(0, 0);
		}
	}
}

#endif // !RIDGE_REGRESSION_HPP