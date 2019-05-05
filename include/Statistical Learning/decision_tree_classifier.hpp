#ifndef DECISION_TREE_CLASSIFIER_HPP
#define DECISION_TREE_CLASSIFIER_HPP

namespace sm {
	namespace tree {

		class DecisionTreeClassifier
		{
		public:
			DecisionTreeClassifier();
			~DecisionTreeClassifier();

			void fit(const nc::NdArray<double>& X, const nc::NdArray<double>& Y);
			std::vector<nc::NdArray<double>> split_dataset(const nc::NdArray<double>& X, const nc::NdArray<double>& Y, int axis, double value);
			nc::NdArray<double> predict(const nc::NdArray<double>& X);
		private:
			Node* root;
			void build_tree(const nc::NdArray<double>& X, const nc::NdArray<double>& Y, const std::vector<int>& vis, Node* root);
			double classify(const nc::NdArray<double>& row, Node* p);
		};

		DecisionTreeClassifier::DecisionTreeClassifier() {}

		DecisionTreeClassifier::~DecisionTreeClassifier()
		{
		}


	}
}


#endif // !DECISION_TREE_CLASSIFIER_HPP