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
			DecisionTreeClassifier();
			~DecisionTreeClassifier();

			void fit(const nc::NdArray<double>& X, const nc::NdArray<int>& Y);
			nc::NdArray<int> predict(const nc::NdArray<double>& X);

			Node* root_;

		private:
			Node* build_tree(const nc::NdArray<double>& X, const nc::NdArray<int>& Y, std::vector<bool>& visit);
			int classify(const nc::NdArray<double>& x, Node* p);
		};

		DecisionTreeClassifier::DecisionTreeClassifier() {}

		DecisionTreeClassifier::~DecisionTreeClassifier()
		{
		}

		Node* DecisionTreeClassifier::build_tree(const nc::NdArray<double>& X, const nc::NdArray<int>& Y, std::vector<bool>& visit) 
		{
			Node* root = new Node();

			if (double_equal(Y.min()(0, 0), Y.max()(0, 0))) {
				root->is_leaf_ = true;
				root->label_ = Y.min()(0, 0);
				return root;
			}

			double best_ent = DBL_MAX;
			nc::NdArray<double> best_attr_values;

			for (int attr_index = 0; attr_index < visit.size(); ++attr_index) 
			{
				if (!visit[attr_index]) 
				{
					double attr_ent = 0;

					nc::NdArray<double> attr_values = nc::unique(X(X.rSlice(), attr_index));
					for (double attr_value : attr_values) 
					{
						attr_ent += shannon_ent(Y[X(X.rSlice(), attr_index) == attr_value]);
					}

					if (attr_ent < best_ent) 
					{
						root->attr_index_ = attr_index;
						best_ent = attr_ent;
						best_attr_values = attr_values;
					}
				}
			}

			if (root->attr_index_ == -1) {
				root->is_leaf_ = true;
				root->label_ = find_most(values_count(Y));
				return root;
			}

			visit[root->attr_index_] = true;

			for (double attr_value : best_attr_values)
			{
				nc::NdArray<bool> filter = (X(X.rSlice(), root->attr_index_) == attr_value);
				Node* child = build_tree(X[filter], Y[filter], visit);
				child->attr_value_ = attr_value;
				if (child->attr_index_ == -1)
					child->attr_index_ = root->attr_index_;

				root->children_.push_back(child);
			}

			return root;
		}

		int DecisionTreeClassifier::classify(const nc::NdArray<double>& x, Node* p) {
			if (p->is_leaf_)
				return p->label_;

			for (Node* c : p->children_) {
				if (x[p->attr_index_] == c->attr_value_)
					return classify(x, c);
			}

			return -1;
		}

		void DecisionTreeClassifier::fit(const nc::NdArray<double>& X, const nc::NdArray<int>& Y)
		{
			nc::Shape s = X.shape();
			std::vector<bool> visit(s.cols, false);
			root_ = build_tree(X, Y, visit);
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

	}
}

#endif // !DECISION_TREE_CLASSIFIER_HPP