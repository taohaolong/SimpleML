#ifndef LOGISTIC_REGRESSION_HPP
#define LOGISTIC_REGRESSION_HPP

namespace sm {
	namespace linear_model {
		class LogisticRegression
		{
		public:
			nc::NdArray<double> coef_;
			double intercept_;
			int n_iter_;

			LogisticRegression(std::string penalty = "l2", double tol = 0.0001, double lr = 0.01, double C = 1.0, bool fit_intercept = true, double random_state = 0, int max_iter = 100);
			~LogisticRegression();

			void fit(const nc::NdArray<double>& X, const nc::NdArray<double>& y);
			nc::NdArray<double> predict(const nc::NdArray<double>& X);

		private:
			std::string penalty_;
			double tol_;
			double lr_;
			double C_;
			bool fit_intercept_;
			double random_state_;
			int max_iter_;
		};

		LogisticRegression::LogisticRegression(std::string penalty, double tol, double lr, double C, bool fit_intercept, double random_state, int max_iter) :
			penalty_(penalty),
			tol_(tol),
			lr_(lr),
			fit_intercept_(fit_intercept),
			random_state_(random_state),
			max_iter_(max_iter)
		{
		}

		LogisticRegression::~LogisticRegression()
		{
		}

		void LogisticRegression::fit(const nc::NdArray<double> & X, const nc::NdArray<double> & y) {
			//StochasticGradientDescent
			nc::Shape s = X.shape();
			nc::NdArray<double> X_temp;
			nc::NdArray<double> Theta;

			std::srand(random_state_);

			if (fit_intercept_) {
				X_temp = nc::hstack({ nc::ones<double>(s.rows, 1), X });
				Theta = nc::ones<double>(s.cols + 1, 1);
			}
			else {
				X_temp = X;
				Theta = nc::ones<double>(s.cols, 1);
			}

			for (int i = 0; i < max_iter_; ++i) {
				double total_error = 0;
				int rand_index;
				for (int j = 0; j < s.rows; ++j) {

					rand_index = rand() % s.rows;

					double h = sigmoid(nc::dot(X_temp(rand_index, X_temp.cSlice()), Theta)(0, 0));
					double error = y(rand_index, 0) - h;

					Theta += X_temp(rand_index, X_temp.cSlice()).transpose() * lr_ * error;

					total_error += error;
				}

				if (total_error / s.rows < tol_)
					break;

				std::cout << "epoch: " << i + 1 << ", error: " << total_error / s.rows << std::endl;
			}

			if (fit_intercept_) {
				intercept_ = Theta(0, 0);
				coef_ = Theta(Theta.rSlice(1), Theta.cSlice());
			}
			else {
				intercept_ = 0;
				coef_ = Theta;
			}
		}

		nc::NdArray<double> LogisticRegression::predict(const nc::NdArray<double> & X) {
			nc::NdArray<double> Z = nc::dot(X, coef_) + intercept_;
			return sigmoid_vec(Z);
		}
	}
}

#endif