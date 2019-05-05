#ifndef STRUCTURES
#define STRUCTURES

namespace sm {
	namespace tree {
		struct Node
		{
			int attr_index;
			int n_child;
			double value;
			double label;
			bool leaf;
			std::vector<Node*> childs;

			Node() : attr_index(-1), n_child(-1), label(-1), leaf(false) {
			}

			~Node() {
				childs.clear();
			}
		};
	}
}

#endif // !STRUCTURES
