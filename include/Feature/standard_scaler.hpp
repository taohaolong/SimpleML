#ifndef STANDARD_SCALER_HPP
#define STANDARD_SCALER_HPP

namespace sm {
	namespace preprocessing {
		class StandardScaler
		{
		public:
			StandardScaler();
			~StandardScaler();

			void fit(nc::NdArray<double>& X);
			nc::NdArray<double> transform(nc::NdArray<double>& X);
			nc::NdArray<double> fit_transform(nc::NdArray<double>& X);

		private:
			nc::NdArray<double> mean_;
			nc::NdArray<double> var_;
		};

		StandardScaler::StandardScaler()
		{
		}

		StandardScaler::~StandardScaler()
		{
		}

		void StandardScaler::fit(nc::NdArray<double>& X) {
			mean_ = nc::mean(X, nc::Axis::ROW);
			var_ = nc::var(X, nc::Axis::ROW);
		}

		nc::NdArray<double> StandardScaler::transform(nc::NdArray<double>& X) {
			nc::NdArray<double> X_ss = (X - mean_) / var_;

			return X_ss;
		}

		nc::NdArray<double> StandardScaler::fit_transform(nc::NdArray<double> & X) {
			fit(X);
			return transform(X);
		}
	}
}

#endif