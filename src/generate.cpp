/***
  *  Project:
  *
  *  File: generate.cpp
  *  Created: Mar 06, 2013
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#include <iostream>
#include <limits>
#include <fftw3.h>

#include "typedefs.hpp"
#include "wil/image.hpp"

namespace hir {

	bool execute_fftw(unsigned int size, fftw_complex* input, fftw_complex* output) {
		// create fft plan
		fftw_plan plan;
		plan = fftw_plan_dft_2d(size, size, input, output, FFTW_FORWARD, FFTW_ESTIMATE);
		fftw_execute(plan);
		// destroy fft plan
		fftw_destroy_plan(plan);
		return true;
	} // execute_cufft()


	bool compute_fft_mat(unsigned int size, real_t* data, real_t*& outdata) {
		std::cout << "++ compute_fft_mat" << std::endl;

		unsigned int size2 = size * size;
		fftw_complex* mat_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size2);
		fftw_complex* mat_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size2);
		for(unsigned int i = 0; i < size; ++ i) {
			for(unsigned int j = 0; j < size; ++ j) {
				mat_in[size * i + j][0] = (double) (unsigned int) data[size * i + j];
				mat_in[size * i + j][1] = 0.0;
				mat_out[size * i + j][0] = 0.0;
				mat_out[size * i + j][1] = 0.0;
			} // for
		} // for
		// execute fft
		execute_fftw(size, mat_in, mat_out);

		outdata = new (std::nothrow) real_t[size * size];
		if(outdata == NULL) {
			std::cerr << "error: memory allocation failed for outdata" << std::endl;
			return false;
		} // if
		real_t min_val = std::numeric_limits<real_t>::min(), max_val = 0.0;
		for(unsigned int i = 0; i < size; ++ i) {
			for(unsigned int j = 0; j < size; ++ j) {
				real_t val = pow(mat_out[size * i + j][0], 2) + pow(mat_out[size * i + j][1], 2);
				if(val > max_val) max_val = val;
				if(val < min_val) min_val = val;
				// swap quadrants
				unsigned int index = size * ((i + (size >> 1)) % size) + ((j + (size >> 1)) % size);
				outdata[index] = val;
			} // for
		} // for

		fftw_free(mat_out);
		fftw_free(mat_in);

		return true;
	} // compute_fft_mat()

} // namespace hir


int main(int narg, char** args) {
	if(narg < 4 || narg > 5) {
		std::cout << "usage: compute-fft <size> <input image> <output image> [<log>]" << std::endl;
		return 0;
	} // if
	unsigned int size = atoi(args[1]);
	//cv::Mat img = cv::imread(args[2], 0);
	wil::Image img(0, 0, 30, 30, 30);
	if(!img.read(args[2], size, size)) {
		std::cerr << "error: could not open image file '" << args[2] << "'" << std::endl;
		exit(-1);
	} // if
	hir::real_t *img_data = new (std::nothrow) hir::real_t[size * size], *fft_data = NULL;
	img.get_data(img_data);
	//compute_fft_mat(size, img_data, args[3]);
	hir::compute_fft_mat(size, img_data, fft_data);
	delete[] img_data;
	if(fft_data == NULL) {
		std::cerr << "error: failed to compute FFT" << std::endl;
		exit(-1);
	} // if

	wil::Image fftimg(size, size, 30, 30, 30);
	unsigned int compute_log = 0;
	if(narg == 5) compute_log = atoi(args[4]);
	if(compute_log == 0) fftimg.construct_image(fft_data);
	else fftimg.construct_log_image(fft_data);
	std::string str(args[3]);
	fftimg.save(str);

	delete[] fft_data;

	return 0;
} // main()
