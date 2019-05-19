#ifndef DECISION_TREE_CLASSIFIER_HPP
#define DECISION_TREE_CLASSIFIER_HPP

namespace sm {
	namespace tree {

		/*
		* 检测数据集中的每个子项是否属于同一分类:
		* If so return 类标签
		* Else
		*     寻找划分数据集的最好特征
		*     划分数据集
		*     划分分支结点
		*     for每个划分的子集
		*         递归调用函数并增加返回结果到分支结点中
		*     return 分支结点
		*/
		class DecisionTreeClassifier
		{
		public:
			Node* root_;

			DecisionTreeClassifier(std::string criterion = "gini", int max_depth = INT_MAX, int max_features=0);
			~DecisionTreeClassifier();

			void fit(const nc::NdArray<double>& X, const nc::NdArray<int>& Y);
			nc::NdArray<int> predict(const nc::NdArray<double>& X);
			nc::NdArray<int> predict_prob(const nc::NdArray<double>& X);

		private:
			int max_depth_;
			int max_features_;
			int n_classes_;
			std::string criterion_;
			std::vector<int> features_;

			Node* build_tree(const nc::NdArray<double>& X, const nc::NdArray<int>& Y, int depth = 0);
			int classify(const nc::NdArray<double>& x, Node* p);
			std::pair<int, double> criter_split(const nc::NdArray<double>& X, const nc::NdArray<int>& Y);
		};

		DecisionTreeClassifier::DecisionTreeClassifier(std::string criterion, int max_depth, int max_features):
			criterion_(criterion),
			max_depth_(max_depth),
			max_features_(max_features)
		{

		}

		DecisionTreeClassifier::~DecisionTreeClassifier()
		{
			delete root_;
			features_.clear();
			criterion_.clear();
		}

		std::pair<int, double> DecisionTreeClassifier::criter_split(const nc::NdArray<double>& X, const nc::NdArray<int>& Y)
		{
			double best_criter = DBL_MAX;
			int best_attr_index = -1;
			double best_attr_value = -1;

			for (int attr_index: features_)
			{
				nc::NdArray<double> attr_values = nc::unique(X(X.rSlice(), attr_index));
				for (double attr_value : attr_values)
				{
					double attr_criter = DBL_MAX;
					if (criterion_ == "entropy")
					{
						attr_criter = shannon_ent(Y[X(X.rSlice(), attr_index) == attr_value]) + shannon_ent(Y[X(X.rSlice(), attr_index) != attr_value]);
					}
					else
					{
						attr_criter = gini(Y[X(X.rSlice(), attr_index) == attr_value]) + gini(Y[X(X.rSlice(), attr_index) != attr_value]);
					}

					if (attr_criter < best_criter)
					{
						best_attr_index = attr_index;
						best_criter = attr_criter;
						best_attr_value = attr_value;
					}
				}	
			}

			return std::pair<int, double>(best_attr_index, best_attr_value);
		}

		Node* DecisionTreeClassifier::build_tree(const nc::NdArray<double>& X, const nc::NdArray<int>& Y, int depth)
		{

			Node* root = new Node();

			if (double_equal(Y.min()[0], Y.max()[0]) ||  depth >= max_depth_) {
				root->label_ = find_most(values_count(Y));
				return root;
			}

			std::pair<int, double> tmp = criter_split(X, Y);

			int best_attr_index = tmp.first;
			double best_attr_value = tmp.second;

			nc::NdArray<bool> left_filter = (X(X.rSlice(), best_attr_index) == best_attr_value);
			nc::NdArray<bool> right_filter = (X(X.rSlice(), best_attr_index) != best_attr_value);

			root->is_leaf_ = false;
			root->attr_index_ = best_attr_index;
			root->attr_value_ = best_attr_value;

			root->left_ = build_tree(X[left_filter], Y[left_filter], depth + 1);
			root->right_ = build_tree(X[right_filter], Y[right_filter], depth + 1);

			return root;
		}

		int DecisionTreeClassifier::classify(const nc::NdArray<double>& x, Node* root) {
			if (root->is_leaf_)
				return root->label_;

			return x[root->attr_index_] == root->attr_value_ ? classify(x, root->left_): classify(x, root->right_);
		}

		void DecisionTreeClassifier::fit(const nc::NdArray<double>& X, const nc::NdArray<int>& Y)
		{
			nc::Shape s = X.shape();
			features_.resize(s.cols);
			std::iota(features_.begin(), features_.end(), 0);
			if (max_features_ > 0 && max_features_ < s.cols)
			{
				features_ = sample(features_, max_features_);
			}

			n_classes_ = nc::unique(Y).size();

			root_ = build_tree(X, Y);
		}

		nc::NdArray<int> DecisionTreeClassifier::predict(const nc::NdArray<double>& X)
		{
			nc::Shape s = X.shape();
			nc::NdArray<int> Y_predict(s.rows, 1);
			for (int i = 0; i < s.rows; ++i) {
				Y_predict[i] = classify(X(i, X.cSlice()), root_);
			}

			return Y_predict;
		}

		nc::NdArray<int> DecisionTreeClassifier::predict_prob(const nc::NdArray<double>& X)
		{
			nc::Shape s = X.shape();
			nc::NdArray<int> Y_predict(s.rows, n_classes_);
			for (int i = 0; i < s.rows; ++i) {
				Y_predict(i, classify(X(i, X.cSlice()), root_)) = 1;
			}

			return Y_predict;
		}
	}
}

#endif // !DECISION_TREE_CLASSIFIER_HPP