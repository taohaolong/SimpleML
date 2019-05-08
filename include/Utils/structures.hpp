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
			std::vector<Node*> children_;

			Node() : 
				attr_index_(-1),
				attr_value_(-1),
				label_(-1), 
				is_leaf_(false) 
			{

			}

			~Node() {
				children_.clear();
			}
		};

		void print_tree(Node* p, int depth = 0) {
			for (int i = 0; i < depth; ++i)
				std::cout << "└───";

			std::cout << "attr_index: " << p->attr_index_ << ", attr_value: " << p->attr_value_;

			if (p->is_leaf_) {
				std::cout << ", lable: " << p->label_ << std::endl;
				return;
			}

			std::cout << std::endl;

			for (Node* c : p->children_)
				print_tree(c, depth + 1);
		}
	}
}

#endif // !STRUCTURES
