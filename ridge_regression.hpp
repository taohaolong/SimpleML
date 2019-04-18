#include "NumCpp.hpp"

using namespace nc;

namespace linear_model {
	class RidgeRegression
	{
	public:
		NdArray<double> coef_;
		double intercept_;

		RidgeRegression(bool fit_intercept = true, double lambda=0.2);
		~RidgeRegression();

		void fit(NdArray<double>& X, NdArray<double>& y);
		NdArray<double> predict(NdArray<double>& X);
		double score(NdArray<double>& X, NdArray<double>& y);

	private:
		bool m_fit_intercept;
		double m_lambda;
	};

	RidgeRegression::RidgeRegression(bool fit_intercept, double lambda):
		m_fit_intercept(fit_intercept),
		m_lambda(lambda)
	{	}

	RidgeRegression::~RidgeRegression()
	{	}

	void RidgeRegression::fit(NdArray<double>& X, NdArray<double>& y) {
		//coef_ = (X^T * X) ^ { -1 } * X^T * Y 
		NdArray<double> X_temp;

		if (m_fit_intercept)
			X_temp = hstack({ ones<double>(X.shape().rows, 1), X });
		else
			X_temp = X;

		NdArray<double> denom = dot(X_temp.transpose(), X_temp) + eye<double>(X_temp.shape().cols);

		NdArray<double> theta = linalg::multi_dot({ linalg::inv(denom), X_temp.transpose(), y });
		
		if (m_fit_intercept) {
			intercept_ = theta(0, 0);
			coef_ = theta(theta.rSlice(1), 0);
		}
		else {
			intercept_ = 0;
			coef_ = theta;
		}

	}

	NdArray<double> RidgeRegression::predict(NdArray<double>& X) {
		return dot(X, coef_) + intercept_;
	}

	double RidgeRegression::score(NdArray<double>& X, NdArray<double>& y) {
		return mean(predict(X) == y)(0, 0);
	}
}