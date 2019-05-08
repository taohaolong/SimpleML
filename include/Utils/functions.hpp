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
		nc::NdArray<double> dist = nc::sum((X - node) * (X - node), nc::Axis::COL);
		dist.reshape(s.rows, 1);

		return dist;
	}

	int find_most(const std::unordered_map<int, int>& m) {
		int res_key = 0;
		double res_value = DBL_MIN;
		for (auto it = m.begin(); it != m.end(); it++) {
			if (it->second > res_value) {
				res_key = it->first;
				res_value = it->second;
			}
		}

		return res_key;
	}

	std::unordered_map<int, int> values_count(const nc::NdArray<int>& data) {
		std::unordered_map<int, int> m;
		for (int i = 0; i < data.size(); ++i)
			if (m.find(data[i]) == m.end())
				m[data[i]] = 1;
			else
				m[data[i]]++;
		return m;
	}

	double shannon_ent(const nc::NdArray<int>& data) {
		nc::Shape s = data.shape();
		std::unordered_map<int, int> label_count = values_count(data);

		double ent = 0, prob = 0;
		for (std::unordered_map<int, int>::iterator it = label_count.begin(); it != label_count.end(); ++it) {
			prob = (double)it->second / s.rows;
			ent -= prob * log2(prob);
		}

		return ent;
	}
}
#endif // !FUNCTIONS_HPP
