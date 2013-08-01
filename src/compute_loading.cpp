/***
  *  Project:
  *
  *  File: compute_loading.cpp
  *  Created: Mar 06, 2013
  *  Modified: Thu 01 Aug 2013 01:23:57 PM PDT
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#include <opencv2/opencv.hpp>
#include "wil/image.hpp"

	double compute_loading(unsigned int size, const cv::Mat& data) {
		unsigned int size2 = size * size;
		unsigned int loading = 0;
		double *temp_data = new (std::nothrow) double[size2];
		for(unsigned int i = 0; i < size; ++ i) {
			for(unsigned int j = 0; j < size; ++ j) {
				double val = (double) (unsigned int) data.at<unsigned char>(i, j);
				if(val > 0.5) {
					++ loading;
					temp_data[size * i + j] = 1.0;
				} else {
					temp_data[size * i + j] = 0.0;
				} // if-else
			} // for
			//std::cout << std::endl;
		} // for

		delete[] temp_data;

		return (double) loading / size2;
	} // compute_fft_mat()


/*int main(int narg, char** args) {
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
} // main()*/


int main(int narg, char** args) {
	if(narg != 3) {
		std::cout << "usage: compute-loading <size> <input image>" << std::endl;
		return 0;
	} // if
	unsigned int size = atoi(args[1]);
	cv::Mat img = cv::imread(args[2], 0);
	std::cout << "Loading factor: " << compute_loading(size, img) << std::endl;
	return 0;
} // main()
