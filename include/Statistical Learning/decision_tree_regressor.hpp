#ifndef DECISION_TREE_REGRESSOR_HPP
#define DECISION_TREE_REGRESSOR_HPP

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
		class DecisionTreeRegressor
		{
		public:
			Node* root_;

			DecisionTreeRegressor(std::string criterion = "mse", int max_depth = INT_MAX, int max_features = 0, double min_impurity_split=1e-3);
			~DecisionTreeRegressor();

			void fit(const nc::NdArray<double>& X, const nc::NdArray<double>& Y);
			nc::NdArray<double> predict(const nc::NdArray<double>& X);

		private:
			int max_depth_;
			int max_features_;
			double min_impurity_split_;
			std::string criterion_;
			std::vector<int> features_;

			Node* build_tree(const nc::NdArray<double>& X, const nc::NdArray<double>& Y, int depth = 0);
			double classify(const nc::NdArray<double>& x, Node* p);
			std::pair<int, double> criter_split(const nc::NdArray<double>& X, const nc::NdArray<double>& Y);
		};

		DecisionTreeRegressor::DecisionTreeRegressor(std::string criterion, int max_depth, int max_features, double min_impurity_split):
			criterion_(criterion),
			max_depth_(max_depth),
			max_features_(max_features),
			min_impurity_split_(min_impurity_split)
		{

		}

		DecisionTreeRegressor::~DecisionTreeRegressor()
		{
			delete root_;
			features_.clear();
			criterion_.clear();
		}

		std::pair<int, double> DecisionTreeRegressor::criter_split(const nc::NdArray<double>& X, const nc::NdArray<double>& Y)
		{
			double best_criter = DBL_MAX;
			int best_attr_index = -1;
			double best_attr_value = -1;

			for (int attr_index : features_)
			{
				nc::NdArray<double> attr_values = nc::unique(X(X.rSlice(), attr_index));
				for (double attr_value : attr_values)
				{
					double attr_criter = DBL_MAX;
					if (criterion_ == "mse")
					{
						attr_criter = mse(Y[X(X.rSlice(), attr_index) <= attr_value]) + mse(Y[X(X.rSlice(), attr_index) > attr_value]);
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

		Node* DecisionTreeRegressor::build_tree(const nc::NdArray<double> & X, const nc::NdArray<double> & Y, int depth)
		{

			Node* root = new Node();

			if (mse(Y) / Y.size() < min_impurity_split_ || depth >= max_depth_) {
				root->label_ = nc::mean(Y, nc::Axis::ROW)[0];
				return root;
			}

			std::pair<int, double> tmp = criter_split(X, Y);

			int best_attr_index = tmp.first;
			double best_attr_value = tmp.second;

			nc::NdArray<bool> left_filter = (X(X.rSlice(), best_attr_index) <= best_attr_value);
			nc::NdArray<bool> right_filter = (X(X.rSlice(), best_attr_index) > best_attr_value);

			root->is_leaf_ = false;
			root->attr_index_ = best_attr_index;
			root->attr_value_ = best_attr_value;

			root->left_ = build_tree(X[left_filter], Y[left_filter], depth + 1);
			root->right_ = build_tree(X[right_filter], Y[right_filter], depth + 1);

			return root;
		}

		double DecisionTreeRegressor::classify(const nc::NdArray<double> & x, Node * root) {
			if (root->is_leaf_)
				return root->label_;

			return x[root->attr_index_] <= root->attr_value_ ? classify(x, root->left_) : classify(x, root->right_);
		}

		void DecisionTreeRegressor::fit(const nc::NdArray<double> & X, const nc::NdArray<double> & Y)
		{
			nc::Shape s = X.shape();
			features_.resize(s.cols);
			std::iota(features_.begin(), features_.end(), 0);
			if (max_features_ > 0 && max_features_ < s.cols)
			{
				features_ = sample(features_, max_features_);
			}

			root_ = build_tree(X, Y);
		}

		nc::NdArray<double> DecisionTreeRegressor::predict(const nc::NdArray<double> & X)
		{
			nc::Shape s = X.shape();
			nc::NdArray<double> Y_predict(s.rows, 1);
			for (int i = 0; i < s.rows; ++i) {
				Y_predict[i] = classify(X(i, X.cSlice()), root_);
			}

			return Y_predict;
		}
	}
}

#endif // !DECISION_TREE_REGRESSOR_HPP