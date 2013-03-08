/***
  *  Project:
  *
  *  File: generate.cpp
  *  Created: Mar 06, 2013
  *  Modified: Thu 07 Mar 2013 09:15:08 AM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#include <opencv2/opencv.hpp>
#include <fftw3.h>
#include "image.hpp"

	bool execute_fftw(unsigned int size, fftw_complex* input, fftw_complex* output) {
		// create fft plan
		fftw_plan plan;
		plan = fftw_plan_dft_2d(size, size, input, output, FFTW_FORWARD, FFTW_ESTIMATE);
		fftw_execute(plan);
		// destroy fft plan
		fftw_destroy_plan(plan);
		return true;
	} // execute_cufft()


	bool compute_fft_mat(unsigned int size, const cv::Mat& data) {
		std::cout << "++ compute_fft_mat" << std::endl;

		unsigned int size2 = size * size;
		fftw_complex* mat_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size2);
		fftw_complex* mat_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size2);
		for(unsigned int i = 0; i < size; ++ i) {
			for(unsigned int j = 0; j < size; ++ j) {
				mat_in[size * i + j][0] = (double) (unsigned int) data.at<unsigned char>(i, j);
				mat_in[size * i + j][1] = 0.0;
				//std::cout << (double) (unsigned int) data.at<unsigned char>(i, j) << ","
				//			<< mat_in[size * i + j][0] << "," << mat_in[size * i + j][1] << " ";
				mat_out[size * i + j][0] = 0.0;
				mat_out[size * i + j][1] = 0.0;
			} // for
			//std::cout << std::endl;
		} // for
		//print_fftwcmatrix("mat_in", mat_in, size_, size_);
		// execute fft
		execute_fftw(size, mat_in, mat_out);

		float *temp_data = new (std::nothrow) float[size * size];
		float min_val = FLT_MAX, max_val = 0.0;
		for(unsigned int i = 0; i < size; ++ i) {
			for(unsigned int j = 0; j < size; ++ j) {
				float val = pow(mat_out[size * i + j][0], 2) + pow(mat_out[size * i + j][1], 2);
				std::cout << mat_out[size * i + j][0] << "," << mat_out[size * i + j][1] << " ";
				if(val > max_val) max_val = val;
				if(val < min_val) min_val = val;
				temp_data[size * i + j] = val;
				//std::cout << val << " ";
			} // for
			std::cout << std::endl;
		} // for

		hig::Image img(size, size);
		img.construct_image(temp_data);
		std::string str("mysquare_fft.tif");
		img.save(str);

		delete[] temp_data;

		//print_fftwcmatrix("mat_out", mat_out, size_, size_);
		//print_cmatrix("f_mat_[f_mat_i_]", f_mat_[f_mat_i_].data(), size_, size_);

		fftw_free(mat_out);
		fftw_free(mat_in);
		return true;
	} // compute_fft_mat()


int main(int narg, char** args) {
	unsigned int size = 512;
	unsigned int size8 = size >> 3;
	//std::vector<unsigned char> data;
	cv::Mat img = cv::Mat::ones(size, size, CV_8U) * 255;

	for(unsigned int i = 0; i < size; ++ i) {
		for(unsigned int j = 0; j < size; ++ j) {
			//unsigned char temp = 255;
			if(i < 3 * size8 || i > 5 * size8) img.at<unsigned char>(i, j) = 0; //temp = 0;
			if(j < 3 * size8 || j > 5 * size8) img.at<unsigned char>(i, j) = 0; //temp = 0;
			//std::cout << temp << " ";
			//data.push_back(temp);
		} // for
	//	std::cout << std::endl;
	} // for
	//cv::imdecode(data, CV_LOAD_IMAGE_GRAYSCALE, &img);
	cv::imwrite("mysquare.tif", img);

	compute_fft_mat(size, img);

	return 0;
} // main()
