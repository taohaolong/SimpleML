#ifndef RANDOM_FOREST_CLASSIFIER
#define RANDOM_FOREST_CLASSIFIER

namespace sm {
	namespace ensemble{
		class RandomForestClassifier
		{
		public:
			std::vector<tree::DecisionTreeClassifier*> estimators_;

			RandomForestClassifier(int n_estimators = 10, int max_depth = INT_MAX, int max_features = 2, std::string criterion = "gini");
			~RandomForestClassifier();

			void fit(const nc::NdArray<double>& X, const nc::NdArray<int>& Y);
			nc::NdArray<int> predict(const nc::NdArray<double>& X);

		private:
			int max_depth_;
			int n_classes_;
			int n_estimators_;
			int max_features_;
			std::string criterion_;
		};

		RandomForestClassifier::RandomForestClassifier(int n_estimators, int max_depth, int max_features, std::string criterion):
			n_estimators_(n_estimators),
			criterion_(criterion),
			max_depth_(max_depth),
			max_features_(max_features)
		{
		}

		RandomForestClassifier::~RandomForestClassifier()
		{
			estimators_.clear();
		}

		void RandomForestClassifier::fit(const nc::NdArray<double>& X, const nc::NdArray<int>& Y) {
			n_classes_ = nc::unique(Y).size();
			for (int n = 0; n < n_estimators_; ++n)
			{
				tree::DecisionTreeClassifier* dtc = new tree::DecisionTreeClassifier(criterion_, max_depth_, max_features_);
				dtc->fit(X, Y);
				estimators_.push_back(dtc);
			}
		}

		nc::NdArray<int> RandomForestClassifier::predict(const nc::NdArray<double>& X) {
			nc::Shape s = X.shape();
			nc::NdArray<int> Y_prob = nc::zeros<int>(s.rows, n_classes_);
			for (int n = 0; n < n_estimators_; ++n)
			{
				Y_prob += estimators_[n]->predict_prob(X);
			}

			nc::NdArray<int> Y_predict = nc::argmax(Y_prob, nc::Axis::COL).astype<int>();

			Y_predict.reshape(s.rows, 1);

			return Y_predict;
		}
	}
}

#endif // !RANDOM_FOREST_CLASSIFIER
