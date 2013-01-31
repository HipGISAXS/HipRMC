
	// temp, for testing
	template <typename value_type>
	void print_matrix(const char* name, value_type* matrix, unsigned int rows, unsigned int cols) {
		std::cout << "++++ " << name << std::endl;
		for(unsigned int i = 0; i < rows; ++ i) {
			for(unsigned int j = 0; j < cols; ++ j) {
				std::cout << matrix[cols * i + j] << "\t";
			} // for
			std::cout << std::endl;
		} // for
	} // print_matrix()

	template <typename value_type>
	void print_cmatrix(const char* name, value_type* matrix, unsigned int rows, unsigned int cols) {
		std::cout << "++++ " << name << std::endl;
		for(unsigned int i = 0; i < rows; ++ i) {
			for(unsigned int j = 0; j < cols; ++ j) {
				std::cout << matrix[cols * i + j].real() << "," << matrix[cols * i + j].imag() << "\t";
			} // for
			std::cout << std::endl;
		} // for
	} // print_matrix()

	template <typename value_type>
	void print_cucmatrix(const char* name, value_type* matrix, unsigned int rows, unsigned int cols) {
		std::cout << "++++ " << name << std::endl;
		for(unsigned int i = 0; i < rows; ++ i) {
			for(unsigned int j = 0; j < cols; ++ j) {
				std::cout << matrix[cols * i + j].x << "," << matrix[cols * i + j].y << "\t";
			} // for
			std::cout << std::endl;
		} // for
	} // print_matrix()

	template <typename value_type>
	void print_fftwcmatrix(const char* name, value_type* matrix, unsigned int rows, unsigned int cols) {
		std::cout << "++++ " << name << std::endl;
		for(unsigned int i = 0; i < rows; ++ i) {
			for(unsigned int j = 0; j < cols; ++ j) {
				std::cout << matrix[cols * i + j][0] << "," << matrix[cols * i + j][1] << "\t";
			} // for
			std::cout << std::endl;
		} // for
	} // print_matrix()

	template <typename value_type>
	void print_array(const char* name, value_type* array, unsigned int len) {
		std::cout << "++++ " << name << std::endl;
		for(unsigned int i = 0; i < len; ++ i)
			std::cout << array[i] << "\t";
		std::cout << std::endl;
	} // print_array()

	template <typename value_type>
	void print_carray(const char* name, value_type* array, unsigned int len) {
		std::cout << "++++ " << name << std::endl;
		for(unsigned int i = 0; i < len; ++ i)
			std::cout << array[i].real() << "," << array[i].imag() << "\t";
		std::cout << std::endl;
	} // print_array()
