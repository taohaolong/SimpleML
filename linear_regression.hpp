#include "NumCpp.hpp"

using namespace nc;

namespace linear_model {
	class LinearRegression {
	public:
		NdArray<double> coef_;
		double intercept_;

		LinearRegression(bool fit_intercept = true);
		~LinearRegression();

		void fit(NdArray<double>& X, NdArray<double>& y);
		NdArray<double> predict(NdArray<double>& X);
		double score(NdArray<double>& X, NdArray<double>& y);

	private:
		bool m_fit_intercept;
	};

	LinearRegression::LinearRegression(bool fit_intercept) :
		m_fit_intercept(fit_intercept)
	{	}

	LinearRegression::~LinearRegression() 
	{

	}

	void LinearRegression::fit(NdArray<double>& X, NdArray<double>& y) {
		//coef_ = (X^T * X) ^ { -1 } * X^T * Y 
		Shape shape = X.shape();
		NdArray<double> X_temp;

		if (m_fit_intercept) 
			X_temp = hstack({ ones<double>(shape.rows, 1), X });
		else
			X_temp = X;

		NdArray<double> xTx = dot(X_temp.transpose(), X_temp);

		if (abs(linalg::det(xTx)) <= (1e-6)) 
			throw "This matrix is singular, cannot do inverse";

		NdArray<double> theta = linalg::multi_dot({ linalg::inv(xTx), X_temp.transpose(), y });

		if (m_fit_intercept) {
			intercept_ = theta(0, 0);
			coef_ = theta(theta.rSlice(1), 0);
		}
		else {
			intercept_ = 0;
			coef_ = theta;
		}
		
	}

	NdArray<double> LinearRegression::predict(NdArray<double>& X) {
		return dot(X, coef_) + intercept_;;
	}

	double LinearRegression::score(NdArray<double>& X, NdArray<double>& y) {
		return mean(predict(X) == y)(0, 0);
	}

}
