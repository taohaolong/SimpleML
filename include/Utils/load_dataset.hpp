#ifndef LOAD_DATASET
#define LOAD_DATASET

namespace sm {
	template<typename dtype>
	nc::NdArray<dtype> load_dataset(const std::string& filename, const std::string& sep="\t") {
		std::ifstream ifs(filename, ios::in);

		if (!ifs.is_open())
			throw "File open failed!";
		else 
		{
			int rows = 0;
			std::vector<dtype> values;

			while (!ifs.eof())
			{
				std::string line;
				std::getline(ifs, line);

				std::vector<std::string> fields;
				boost::split(fields, line, boost::algorithm::is_any_of(sep));
					
				for(std::string field: fields)
					values.push_back(static_cast<dtype>(std::stod(field)));
				rows++;
			}

			ifs.close();

			if (values.size() % rows != 0) 
			{
				throw "The number of column in each row is unequal!";
			}
			else 
			{
				int cols = values.size() / rows;
				nc::NdArray<dtype> data(rows, cols);
				for (int row = 0; row < rows; ++row) {
					for (int col = 0; col < cols; ++col)
						data(row, col) = values[row * cols + col];
				}
				
				return data;
			}
		}
	}
}

#endif // !LOAD_DATASET