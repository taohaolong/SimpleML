#ifndef MIN_MAX_SCALER_HPP
#define MIN_MAX_SCALER_HPP

namespace sm {
	namespace preprocessing {
		class MinMaxScaler
		{
		public:
			MinMaxScaler();
			~MinMaxScaler();

			void fit(nc::NdArray<double>& X);
			nc::NdArray<double> transform(nc::NdArray<double>& X);
			nc::NdArray<double> fit_transform(nc::NdArray<double>& X);

		private:
			nc::NdArray<double> min_;
			nc::NdArray<double> max_;
		};

		MinMaxScaler::MinMaxScaler()
		{
		}

		MinMaxScaler::~MinMaxScaler()
		{
		}

		void MinMaxScaler::fit(nc::NdArray<double>& X) {
			min_ = nc::min(X, nc::Axis::ROW);
			max_ = nc::max(X, nc::Axis::ROW);
		}

		nc::NdArray<double> MinMaxScaler::transform(nc::NdArray<double>& X) {
			nc::NdArray<double> X_mm = (X - min_) / max_;

			return X_mm;
		}

		nc::NdArray<double> MinMaxScaler::fit_transform(nc::NdArray<double> & X) {
			fit(X);
			return transform(X);
		}
	}
}

#endif