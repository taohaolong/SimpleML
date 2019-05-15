#ifndef SIMPLEML_HPP
#define SIMPLEML_HPP

#include <cmath>
#include <random>
#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <numeric>
#include <unordered_map>
#include <boost/algorithm/string.hpp>

#include "NumCpp.hpp"

namespace sm {
	struct version
	{
		static const unsigned int major = 0;
		static const unsigned int minor = 1;
		static const unsigned int patch = 0;

		static inline std::string as_string()
		{

			std::stringstream ss;
			ss << version::major << '.' << version::minor << '.' << version::patch;

			return ss.str();
		}
	};
}

#ifdef _MSC_VER
#pragma warning(disable : 4018)
#endif

//工具箱
#include "Utils/load_dataset.hpp"
#include "Utils/constants.hpp"
#include "Utils/functions.hpp"
#include "Utils/structures.hpp"

//特征工程
#include "Feature/standard_scaler.hpp"

//统计学习
#include "Statistical Learning/decision_tree_classifier.hpp"
#include "Statistical Learning/decision_tree_regressor.hpp"
#include "Statistical Learning/gaussian_mixture.hpp"
#include "Statistical Learning/k_means.hpp"
#include "Statistical Learning/k_neighbors_classifier.hpp"
#include "Statistical Learning/k_neighbors_regressor.hpp"
#include "Statistical Learning/linear_regression.hpp"
#include "Statistical Learning/logistic_regression.hpp"
#include "Statistical Learning/ridge_regression.hpp"

#endif