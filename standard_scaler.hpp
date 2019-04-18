#include "NumCpp.hpp"
using namespace nc;

namespace preprocessing {
	class StandardScaler
	{
	public:
		StandardScaler();
		~StandardScaler();

		void fit(NdArray<double>& X);
		NdArray<double> transform(NdArray<double>& X);
		NdArray<double> fit_transform(NdArray<double>& X);

	private:
		NdArray<double> m_X_mean;
		NdArray<double> m_X_var;
	};

	StandardScaler::StandardScaler()
	{
	}

	StandardScaler::~StandardScaler()
	{
	}

	void StandardScaler::fit(NdArray<double>& X) {
		m_X_mean = mean(X, Axis::ROW);
		m_X_var = var(X, Axis::ROW);
	}

	NdArray<double> StandardScaler::transform(NdArray<double>& X) {
		Shape shape = X.shape();
		NdArray<double> X_SS(shape.rows, shape.cols);
		for (int i = 0; i < shape.rows; i++)
			for (int j = 0; j < shape.cols; ++j)
				if(abs(m_X_var(1, j)) < (1e-6))
					X_SS(i, j) = X(i, j) / shape.rows;
				else
					X_SS(i, j) = (X(i, j) - m_X_mean(0, j)) / m_X_var(0, j);

		return X_SS;
	}

	NdArray<double> StandardScaler::fit_transform(NdArray<double>& X) {
		fit(X);
		return transform(X);
	}
}