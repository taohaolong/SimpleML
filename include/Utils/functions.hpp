#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP
namespace sm {
	bool double_equal(double a, double b) {
		if (std::abs(a - b) <= (1e-6))
			return true;
		return false;
	}

	double sigmoid(double z) {
		return 1 / (1 + std::exp(-1 * z));
	}

	nc::NdArray<double> sigmoid_vec(const nc::NdArray<double>& Z) {
		nc::Shape s = Z.shape();
		nc::NdArray<double> h(s);
		for (int i = 0; i < s.rows; ++i) {
			for (int j = 0; j < s.cols; ++j) {
				h(i, j) = sigmoid(Z(i, j));
			}
		}

		return h;
	}

	nc::NdArray<double> euclidean(const nc::NdArray<double>& node, const nc::NdArray<double>& X) {
		nc::Shape s = X.shape();
		nc::NdArray<double> dis = nc::sum((X - node) * (X - node), nc::Axis::COL);
		dis.reshape(s.rows, 1);

		return dis;
	}

	double shannon_ent(nc::NdArray<double>& data) {
		nc::Shape s = data.shape();
		std::unordered_map<double, int> label_count;
		for (int i = 0; i < s.rows; ++i)
			if (label_count.find(data(i, 0)) == label_count.end())
				label_count[data(i, 0)] = 1;
			else
				label_count[data(i, 0)]++;

		double ent = 0, prob = 0;
		for (std::unordered_map<double, int>::iterator it = label_count.begin(); it != label_count.end(); ++it) {
			prob = it->second / s.rows;
			ent -= prob * log(prob);
		}

		return ent;
	}
}
#endif // !FUNCTIONS_HPP
