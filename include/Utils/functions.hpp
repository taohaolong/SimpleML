#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP
namespace sm {
	bool double_equal(double a, double b) 
	{
		if (std::abs(a - b) <= (1e-6))
			return true;
		return false;
	}

	double sigmoid(double z) 
	{
		return 1 / (1 + std::exp(-1 * z));
	}

	nc::NdArray<double> sigmoid_vec(const nc::NdArray<double>& Z) 
	{
		nc::Shape s = Z.shape();
		nc::NdArray<double> h(s);
		for (int i = 0; i < s.rows; ++i) 
		{
			for (int j = 0; j < s.cols; ++j) 
			{
				h(i, j) = sigmoid(Z(i, j));
			}
		}

		return h;
	}

	nc::NdArray<double> euclidean(const nc::NdArray<double>& node, const nc::NdArray<double>& X) 
	{
		nc::Shape s = X.shape();
		nc::NdArray<double> dist = nc::sum((X - node) * (X - node), nc::Axis::COL);
		dist.reshape(s.rows, 1);

		return dist;
	}

	int find_most(const std::unordered_map<int, int>& m) 
	{
		int res_key = 0;
		double res_value = DBL_MIN;
		for (auto it = m.begin(); it != m.end(); it++) 
		{
			if (it->second > res_value) 
			{
				res_key = it->first;
				res_value = it->second;
			}
		}

		return res_key;
	}

	std::unordered_map<int, int> values_count(const nc::NdArray<int>& data) 
	{
		std::unordered_map<int, int> m;
		for (int i = 0; i < data.size(); ++i)
			if (m.find(data[i]) == m.end())
				m[data[i]] = 1;
			else
				m[data[i]]++;
		return m;
	}

	double shannon_ent(const nc::NdArray<int>& data) 
	{
		nc::Shape s = data.shape();
		std::unordered_map<int, int> label_count = values_count(data);

		double ent = 0, prob = 0;
		for (std::unordered_map<int, int>::iterator it = label_count.begin(); it != label_count.end(); ++it) 
		{
			prob = (double)it->second / s.rows;
			ent -= prob * log2(prob);
		}

		return ent;
	}

	double gini(const nc::NdArray<int>& data)
	{
		nc::Shape s = data.shape();
		std::unordered_map<int, int> label_count = values_count(data);

		double g = 1, prob = 0;
		for (std::unordered_map<int, int>::iterator it = label_count.begin(); it != label_count.end(); ++it)
		{
			prob = (double)it->second / s.rows;
			g -= prob * prob;
		}

		return g;
	}

	nc::NdArray<double> multivariate_normal(const nc::NdArray<double>& X, const nc::NdArray<double> mu, const nc::NdArray<double> sigma)
	{
		nc::Shape s = X.shape();
		nc::NdArray<double> num = nc::exp( nc::linalg::multi_dot<double>({ (X - mu), nc::linalg::inv(sigma) , (X - mu).transpose() }) * (-0.5) );
		double denom = pow((2 * PI), 0.5 * s.cols) * pow(nc::linalg::det(sigma), 0.5);
		nc::NdArray<double> ret = nc::diagonal(num / denom);
		return ret;
	}

	std::vector<int> sample(const std::vector<int>& v, int num)
	{
		std::vector<int> v_copy(v);
		std::vector<int> ret(num, 0);
		std::random_device rd;
		std::mt19937 g(rd());

		std::shuffle(v_copy.begin(), v_copy.end(), g);
		ret.assign(v_copy.begin(), v_copy.begin() + num);
		std::sort(ret.begin(), ret.end());
		return ret;
	}
}
#endif // !FUNCTIONS_HPP
