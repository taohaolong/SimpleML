#ifndef LinearRegression_HPP
#define LinearRegression_HPP

namespace sm {
	namespace linear_model {
		class LinearRegression {
		public:
			nc::NdArray<double> coef_;
			double intercept_;

			LinearRegression(bool fit_intercept = true);
			~LinearRegression();

			void fit(const nc::NdArray<double>& X, const nc::NdArray<double>& y);
			nc::NdArray<double> predict(const nc::NdArray<double>& X);
			double score(const nc::NdArray<double>& X, const nc::NdArray<double>& y);

		private:
			bool fit_intercept_;
		};

		LinearRegression::LinearRegression(bool fit_intercept) :
			fit_intercept_(fit_intercept)
		{	}

		LinearRegression::~LinearRegression()
		{

		}

		void LinearRegression::fit(const nc::NdArray<double>& X, const nc::NdArray<double>& y) {
			//coef_ = (X^T * X) ^ { -1 } * X^T * Y 
			nc::Shape shape = X.shape();
			nc::NdArray<double> X_temp;

			if (fit_intercept_)
				X_temp = nc::hstack({ nc::ones<double>(shape.rows, 1), X });
			else
				X_temp = X;

			nc::NdArray<double> xTx = nc::dot(X_temp.transpose(), X_temp);

			if (double_equal(nc::linalg::det(xTx), 0))
				throw "This matrix is singular, cannot do inverse";

			nc::NdArray<double> theta = nc::linalg::multi_dot({ nc::linalg::inv(xTx), X_temp.transpose(), y });

			if (fit_intercept_) {
				intercept_ = theta(0, 0);
				coef_ = theta(theta.rSlice(1), 0);
			}
			else {
				intercept_ = 0;
				coef_ = theta;
			}

		}

		nc::NdArray<double> LinearRegression::predict(const nc::NdArray<double>& X) {
			return dot(X, coef_) + intercept_;;
		}

		double LinearRegression::score(const nc::NdArray<double>& X, const nc::NdArray<double>& y) {
			return nc::mean(predict(X) == y)(0, 0);
		}

	}
}

#endif // !LinearRegression
