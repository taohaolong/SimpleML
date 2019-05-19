#ifndef RANDOM_FOREST_REGRESSOR
#define RANDOM_FOREST_REGRESSOR

namespace sm {
	namespace ensemble {
		class DecisionTreeRegressor
		{
		public:
			std::vector<tree::DecisionTreeRegressor*> estimators_;

			DecisionTreeRegressor(int n_estimators = 10, std::string criterion = "mse", int max_depth = INT_MAX, int max_features = 2);
			~DecisionTreeRegressor();

			void fit(const nc::NdArray<double>& X, const nc::NdArray<double>& Y);
			nc::NdArray<double> predict(const nc::NdArray<double>& X);

		private:
			int max_depth_;
			int n_classes_;
			int n_estimators_;
			int max_features_;
			std::string criterion_;
		};

		DecisionTreeRegressor::DecisionTreeRegressor(int n_estimators, std::string criterion, int max_depth, int max_features) :
			n_estimators_(n_estimators),
			criterion_(criterion),
			max_depth_(max_depth),
			max_features_(max_features)
		{
			estimators_.clear();
		}

		DecisionTreeRegressor::~DecisionTreeRegressor()
		{
		}

		void DecisionTreeRegressor::fit(const nc::NdArray<double>& X, const nc::NdArray<double>& Y) {
			n_classes_ = nc::unique(Y).size();
			for (int n = 0; n < n_estimators_; ++n)
			{
				tree::DecisionTreeRegressor* dtr = new tree::DecisionTreeRegressor(criterion_, max_depth_, max_features_);
				dtr->fit(X, Y);
				estimators_.push_back(dtr);
			}
		}

		nc::NdArray<double> DecisionTreeRegressor::predict(const nc::NdArray<double>& X) {
			nc::Shape s = X.shape();
			nc::NdArray<double> Y_predict = nc::zeros<double>(s.cols);
			for (int n = 0; n < n_estimators_; ++n)
			{
				Y_predict += estimators_[n]->predict(X);
			}

			return Y_predict / n_estimators_;
		}
	}
}

#endif // !RANDOM_FOREST_REGRESSOR
