#ifndef STRUCTURES
#define STRUCTURES

namespace sm {
	namespace tree {
		struct Node
		{
			int attr_index_;
			double attr_value_;
			int label_;
			bool is_leaf_;
			Node* left_;
			Node* right_;

			Node() : 
				attr_index_(-1),
				attr_value_(-1),
				label_(-1), 
				is_leaf_(true),
				left_(NULL),
				right_(NULL)
			{

			}

			~Node() {
				delete left_;
				delete right_;
			}
		};

		void print_tree(Node* root, int depth = 0) {
			for (int i = 0; i < depth; ++i)
				std::cout << "└───";

			if (root->is_leaf_) {
				std::cout << "**leaf(lable: " << root->label_ << ")**" << std::endl;
				return;
			}

			std::cout << "node(index: " << root->attr_index_ << ", value: " << root->attr_value_ << ")";

			std::cout << std::endl;

			print_tree(root->left_, depth + 1);
			print_tree(root->right_, depth + 1);
		}
	}
}

#endif // !STRUCTURES
