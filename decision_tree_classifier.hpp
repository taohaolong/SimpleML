#include "NumCpp.hpp"
#include <iostream>
#include <string>
#include <unordered_map>
using namespace nc;
using namespace std;

namespace tree {
	struct Node
	{
		int attr_index;
		int n_child;
		double value;
		double label;
		bool leaf;
		vector<Node*> childs;

		Node(): attr_index(-1), n_child(-1), label(-1), leaf(false) {
		}

		~Node() {
			childs.clear();
		}
	};

	class DecisionTreeClassifier
	{
	public:
		DecisionTreeClassifier();
		~DecisionTreeClassifier();

		void fit(NdArray<double>& X, NdArray<double>& Y);
		vector<NdArray<double>> SplitDataSet(NdArray<double>& X, NdArray<double>& Y, int axis, double value);
		NdArray<double> predict(NdArray<double>& X);
	private:
		Node* root;
		double CalShannonEnt(NdArray<double>& data);
		void BuildTree(NdArray<double>& X, NdArray<double>& Y, vector<int>& vis, Node* root);
		double classify(const NdArray<double>& row, Node* p);
	};

	DecisionTreeClassifier::DecisionTreeClassifier(){}

	DecisionTreeClassifier::~DecisionTreeClassifier()
	{
	}

	double DecisionTreeClassifier::CalShannonEnt(NdArray<double>& data) {
		Shape s = data.shape();
		unordered_map<double, int> label_count;
		for (int i = 0; i < s.rows; ++i)
			if (label_count.find(data(i, 0)) == label_count.end())
				label_count[data(i, 0)] = 1;
			else
				label_count[data(i, 0)]++;

		double ent = 0, prob = 0;
		for (unordered_map<double, int>::iterator it = label_count.begin(); it != label_count.end(); ++it) {
			prob = it->second / s.rows;
			ent -= prob * log(prob);
		}

		return ent;
	}

	vector<NdArray<double>> DecisionTreeClassifier::SplitDataSet(NdArray<double>& X, NdArray<double>&Y, int axis, double value) {
		vector<NdArray<double>> ret;
		Shape s = X.shape();
		NdArray<double> feature = X(X.rSlice(), axis);
		int num = (int)sum(feature[feature == value])(0, 0);
		int j = 0;
		NdArray<double> X_ret(num, s.cols - 1);
		NdArray<double> Y_ret(num, 1);

		for (int i = 0; i < s.rows; ++i) {
			if (X(i, axis) == value) {
				for (int k = 0; k < s.cols; ++k) {
					if (k == axis) continue;
					else if(k < axis)
						X_ret(j, k) = X(i, k);
					else 
						X_ret(j, k - 1) = X(i, k);
				}
				Y_ret(j++, 0) = Y(i, 0);
			}
			cout << X_ret << endl;
			cout << Y_ret << endl;
		}

		ret.push_back(X_ret);
		ret.push_back(Y_ret);
		return ret;
	}

	void DecisionTreeClassifier::BuildTree(NdArray<double>& X, NdArray<double>& Y, vector<int>& vis, Node* root) {
		if (Y.min()(0, 0) == Y.max()(0, 0)) {
			root->label = Y.min()(0, 0);
			root->leaf = true;
		}
		double base_ent = CalShannonEnt(Y);
		double best_gain = 0, best_index = -1;
		for (int i = 0; i < vis.size(); ++i) {
			if (vis[i] == 1)
				continue;

			double cur_ent = 0;
			NdArray<double> unique_values = unique(X(X.rSlice(), i));
			for (int j = 0; j < unique_values.shape().rows; ++j) {
				double value = unique_values(j, 0);
				vector<NdArray<double>> splited_dataset = SplitDataSet(X, Y, i, value);
				cur_ent += CalShannonEnt(splited_dataset[1]);
			}

			if (best_gain < base_ent - cur_ent) {
				best_index = i;
				best_gain = base_ent - cur_ent;
			}
		}
		vis[best_index] = 1;
		
		NdArray<double> values = unique(X(X.rSlice(), best_index));
		root->n_child = values.shape().rows; 
		for (int j = 0; j < root->n_child; ++j) {
			double value = values(j, 0);
			vector<NdArray<double>> splited = SplitDataSet(X, Y, best_index, value);
			Node* child = new Node();
			child->attr_index = best_index;
			child->value = value;
			BuildTree(splited[0], splited[1], vis, child);
			root->childs.push_back(child);
		}
	}

	void DecisionTreeClassifier::fit(NdArray<double>& X, NdArray<double>& Y) {
		Shape s = X.shape();
		vector<int> vis(s.cols, 0);
		root = new Node();
		BuildTree(X, Y, vis, root);
	}

	double DecisionTreeClassifier::classify(const NdArray<double>& row, Node* p) {
		if (p->leaf)
			return p->label;
		for (Node* child : p->childs) {
			if (row(0, p->attr_index) == p->value)
				return classify(row, child);
		}

		throw "存在值不匹配！";
	}

	NdArray<double> DecisionTreeClassifier::predict(NdArray<double>& X) {
		Shape s = X.shape();
		NdArray<double> Y(s.rows, 1);
		for (int i = 0; i < s.rows; ++i) {
			Node* p = root;
			Y(i, 0) = classify(X(i, X.cSlice()), p);
		}
		
		return Y;
	}

}