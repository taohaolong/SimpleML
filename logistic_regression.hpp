#include <iostream>
#include <cmath>
#include "NumCpp.hpp"

using namespace nc;
using namespace std;

namespace linear_model {
	class LogisticRegression
	{
	public:
		NdArray<double> coef_;
		double intercept_;
		int n_iter_;

		LogisticRegression(string penalty = "l2", double tol = 0.0001, double lr = 0.01, double C = 1.0, bool fit_intercept = true, double random_state = 0, int max_iter = 100);
		~LogisticRegression();

		double Sigmoid(double z);
		NdArray<double> SigmoidVec(NdArray<double>& Z);
		void fit(NdArray<double>& X, NdArray<double>& y);
		NdArray<double> predict(NdArray<double>& X);

	private:
		string m_penalty;
		double m_tol;
		double m_lr;
		double m_C;
		bool m_fit_intercept;
		double m_random_state;
		int m_max_iter;
	};

	LogisticRegression::LogisticRegression(string penalty, double tol, double lr, double C, bool fit_intercept, double random_state, int max_iter):
		m_penalty(penalty),
		m_tol(tol),
		m_lr(lr),
		m_fit_intercept(fit_intercept),
		m_random_state(random_state),
		m_max_iter(max_iter)
	{
	}

	LogisticRegression::~LogisticRegression()
	{
	}

	double LogisticRegression::Sigmoid(double z) {
		return 1 / (1 + exp(-1 * z));
	}

	NdArray<double> LogisticRegression::SigmoidVec(NdArray<double>& Z) {
		Shape s = Z.shape();
		NdArray<double> h(s);
		for (int i = 0; i < s.rows; ++i) {
			for (int j = 0; j < s.cols; ++j) {
				h(i, j) = Sigmoid(Z(i, j));
			}
		}

		return h;
	}

	void LogisticRegression::fit(NdArray<double>& X, NdArray<double>& y) {
		//StochasticGradientDescent
		Shape s = X.shape();
		NdArray<double> X_temp;
		NdArray<double> Theta;

		srand(m_random_state);

		if (m_fit_intercept) {
			X_temp = hstack({ ones<double>(s.rows, 1), X });
			Theta = ones<double>(s.cols+1, 1);
		}
		else {
			X_temp = X;
			Theta = ones<double>(s.cols, 1);
		}

		for (int i = 0; i < m_max_iter; ++i) {
			cout << Theta << endl;
			double total_error = 0;
			int rand_index;
			for(int j = 0; j < s.rows; ++j){

				rand_index = rand() % s.rows;
				
				double h = Sigmoid(dot(X_temp(rand_index, X_temp.cSlice()), Theta)(0, 0));
				double error = y(rand_index, 0) - h;

				Theta += X_temp(rand_index, X_temp.cSlice()).transpose() * m_lr * error;

				total_error += error;
			}

			if (total_error / s.rows < m_tol)
				break;

			cout << "epoch: " << i + 1 << ", error: " << total_error / s.rows << endl;
		}

		if (m_fit_intercept) {
			intercept_ = Theta(0, 0);
			coef_ = Theta(Theta.rSlice(1), Theta.cSlice());
		}
		else {
			intercept_ = 0;
			coef_ = Theta;
		}
	}

	NdArray<double> LogisticRegression::predict(NdArray<double>& X) {
		NdArray<double> Z = dot(X, coef_) + intercept_;
		return SigmoidVec(Z);
	}
}