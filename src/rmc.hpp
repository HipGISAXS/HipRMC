/***
  *  Project: HipRMC - High Performance Reverse Monte Carlo Simulations
  *
  *  File: rmc.hpp
  *  Created: Jan 25, 2013
  *  Modified: Thu 07 Mar 2013 03:53:19 PM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef __RMC_HPP__
#define __RMC_HPP__

#include <random>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include "typedefs.hpp"
#include "tile.hpp"
#include "constants.hpp"
#ifdef USE_GPU
#include "init_gpu.cuh"
#endif

namespace hir {

	class RMC {
		private:
			mat_real_t in_pattern_;	// input pattern and related matrix info
			unsigned int rows_;
			unsigned int cols_;
			unsigned int size_;
										// any benefit of using vectors for below instead? ...
			unsigned int* in_mask_;		// the input mask
			unsigned int in_mask_len_;	// size of input mask
			unsigned int* mask_mat_;	// mask matrix of 1 and 0
			unsigned int num_tiles_;	// total number of tiles
			std::vector<Tile> tiles_;	// the tiles -- temp
			mat_complex_t vandermonde_mat_;
			real_t base_norm_;			// norm of input

			// extracts raw data from image
			bool pre_init(const char*, real_t*);
			// initializes with raw data
			bool init(real_t*, unsigned int, unsigned int*, real_t*);
			bool scale_image_colormap(cv::Mat&, double, double);

		public:
			RMC(unsigned int, unsigned int, const char*, unsigned int, real_t*);
			~RMC();
			bool simulate(int, real_t, unsigned int);

			// for testing ...
			bool scale(unsigned int size);
	}; // class RMC


	RMC::RMC(unsigned int rows, unsigned int cols, const char* img_file,
					unsigned int num_tiles, real_t* loading) :
		in_pattern_(rows, cols),
		rows_(rows), cols_(cols), size_(std::max(rows, cols)),
		in_mask_(NULL),
		in_mask_len_(0),
		mask_mat_(NULL),
		num_tiles_(num_tiles),
		vandermonde_mat_(std::max(rows, cols), std::max(rows, cols)) {

		// for now only square patterns are considered
		if(rows_ != cols_) {
			std::cerr << "error: number of rows should equal number of columns" << std::endl;
			exit(1);
		} // if
		if(!pre_init(img_file, loading)) {
			std::cerr << "error: failed to pre-initialize RMC object" << std::endl;
			exit(1);
		} // if
	} // RMC::RMC()


	RMC::~RMC() {
		if(in_mask_ != NULL) delete[] in_mask_;
		if(mask_mat_ != NULL) delete[] mask_mat_;
		//if(tiles_ != NULL) delete[] tiles_;
	} // RMC::~RMC()


	// idea is that this can be replaced easily for other types of raw inputs (not image)
	bool RMC::pre_init(const char* img_file, real_t* loading) {
		//std::cout << "++ pre_init" << std::endl;
		// TODO: opencv usage is temporary. improve with something else...
		cv::Mat img = cv::imread(img_file, 0);	// grayscale only for now
		//cv::getRectSubPix(img, cv::Size(rows_, cols_), cv::Point2f(cx, cy), subimg);
		// extract the input image raw data (grayscale values)
		// and create mask array = indices in image data where value is 0
		real_t *img_data = new (std::nothrow) real_t[rows_ * cols_];
		unsigned int *mask_data = new (std::nothrow) unsigned int[rows_ * cols_];
		unsigned int mask_count = 0;
		unsigned int hrow = rows_ >> 1;
		unsigned int hcol = cols_ >> 1;
		double min_val, max_val;
		cv::minMaxIdx(img, &min_val, &max_val);
		double threshold = min_val;// + 2 * ceil(max_val / (min_val + 1));
		std::cout << "MIN: " << min_val << ", MAX: " << max_val << ", THRESH: " << threshold << std::endl;
		cv::threshold(img, img, threshold, max_val, cv::THRESH_TOZERO);
		// scale pixel intensities to span all of 0 - 255
		scale_image_colormap(img, threshold, max_val);
		cv::imwrite("hohohohohohoho.tif", img);
		for(unsigned int i = 0; i < rows_; ++ i) {
			for(unsigned int j = 0; j < cols_; ++ j) {
				unsigned int temp = (unsigned int) img.at<unsigned char>(i, j);
				/*
				// do the quadrant swap thingy ...
				unsigned int img_index = cols_ * ((i + hrow) % rows_) + (j + hcol) % cols_;*/
				// or not ...
				unsigned int img_index = cols_ * i + j;
				img_data[img_index] = (real_t) temp;
				if(temp == 0) mask_data[mask_count ++] = img_index;
			} // for
		} // for

		//print_matrix("img_data:", img_data, rows_, cols_);
		//print_array( "mask_data:", mask_data, mask_count);

		#ifdef USE_GPU
			init_gpu();
		#endif

		// TODO: take a subimage of the input ...

		if(!init(img_data, mask_count, mask_data, loading)) {
			std::cerr << "error: failed to initialize RMC object" << std::endl;
			delete[] mask_data;
			delete[] img_data;
			return false;
		} // if

		delete[] mask_data;
		delete[] img_data;
		return true;
	} // RMC::pre_init()


	bool RMC::scale_image_colormap(cv::Mat& img, double min_val, double max_val) {
		for(unsigned int i = 0; i < rows_; ++ i) {
			for(unsigned int j = 0; j < cols_; ++ j) {
				unsigned char temp = img.at<unsigned char>(i, j);
				if(temp != 0) {
					temp = (unsigned char) 255 * (temp - min_val) / (max_val - min_val);
					//std::cout << (unsigned int) temp << " ";
					img.at<unsigned char>(i, j) = temp;
				} // if
			} // for
			//std::cout << std::endl;
		} // for
		return true;
	} // RMC::scale_image_colormap()


	// initialize with raw data
	bool RMC::init(real_t* pattern, unsigned int mask_len, unsigned int* mask, real_t* loading) {
		woo::BoostChronoTimer mytimer;

		//std::cout << "++ init" << std::endl;
		unsigned int size2 = size_ * size_;

		in_pattern_.populate(pattern);
		// create mask and loading arays
		mytimer.start();
		in_mask_len_ = mask_len;
		in_mask_ = new (std::nothrow) unsigned int[mask_len];
		if(in_mask_ == NULL) return false;
		memcpy(in_mask_, mask, mask_len * sizeof(unsigned int));
		// generate mask matrix
		mask_mat_ = new (std::nothrow) unsigned int[size2];
		for(unsigned int i = 0; i < size2; ++ i) mask_mat_[i] = 1;
		for(unsigned int i = 0; i < in_mask_len_; ++ i) mask_mat_[in_mask_[i]] = 0;
		mytimer.stop();
		std::cout << "**** Mask creation time: " << mytimer.elapsed_msec() << " ms." << std::endl;
		//print_matrix("mask_mat:", mask_mat_, size_, size_);

		// compute base norm
		mytimer.start();
		base_norm_ = 0.0;		// why till size/2 only ??? and what is Y ???
		unsigned int maxi = size_;// >> 1;
		for(unsigned int i = 0; i < maxi; ++ i) {
			for(unsigned int j = 0; j < maxi; ++ j) {
				base_norm_ += in_pattern_(i, j); // * (j + 1);	// skipping creation of Y matrix
			} // for
		} // for
		mytimer.stop();
		std::cout << "**** Base norm time: " << mytimer.elapsed_msec() << " ms." << std::endl;

		//std::cout << "++ base_norm: " << base_norm_ << std::endl;

		// create array of random indices
		mytimer.start();
		/*std::vector<unsigned int> indices;
		for(unsigned int i = 0; i < size2; ++ i) indices.push_back(i);
		std::random_shuffle(indices.begin(), indices.end());*/
		std::vector<unsigned int> indices;
		for(unsigned int i = 0; i < size2; ++ i) indices.push_back(i);
		// using mersenne-twister
		std::random_device rd;
		std::mt19937_64 gen(rd());
		std::shuffle(indices.begin(), indices.end(), gen);
		std::cout << "**** Indices creation time: " << mytimer.elapsed_msec() << " ms." << std::endl;
		mytimer.stop();
		//print_array("indices", (unsigned int*)&indices[0], indices.size());

		// compute vandermonde matrix
		mytimer.start();
		// generate 1st order power (full 360 deg rotation in polar coords)
		std::vector<complex_t> first_pow;
		for(unsigned int i = 0; i < size_; ++ i) {
			real_t temp = 2.0 * PI_ * (1.0 - ((real_t)i / size_));
			real_t temp_r = cos(temp);
			real_t temp_i = sin(temp);
			//temp_r = abs(temp_r) < ZERO_LIMIT_ ? 0.0 : temp_r;
			//temp_i = abs(temp_i) < ZERO_LIMIT_ ? 0.0 : temp_i;
			first_pow.push_back(complex_t(temp_r, temp_i));
		} // for
		//print_carray("first_pow", reinterpret_cast<complex_t*>(&first_pow[0]), size_);
		// initialize first column
		typename mat_complex_t::col_iterator citer = vandermonde_mat_.column(0);
		for(unsigned int i = 0; i < citer.size(); ++ i) citer[i] = complex_t(1.0, 0.0);
		// compute rest of the matrix
		typename mat_complex_t::col_iterator curr_citer = vandermonde_mat_.begin_col();
		typename mat_complex_t::col_iterator prev_citer = vandermonde_mat_.begin_col();
		++ curr_citer;
		for(; curr_citer != vandermonde_mat_.end_col(); ++ curr_citer, ++ prev_citer) {
			for(unsigned int i = 0; i < size_; ++ i) curr_citer[i] = prev_citer[i] * first_pow[i];
		} // while
		mytimer.stop();
		std::cout << "**** Vandermonde time: " << mytimer.elapsed_msec() << " ms." << std::endl;
		//print_cmatrix("vandermonde_mat", vandermonde_mat_.data(), size_, size_);

		// initialize tiles
		mytimer.start();
		for(unsigned int i = 0; i < num_tiles_; ++ i)
			tiles_.push_back(Tile(size_, size_, indices));
		mytimer.stop();
		std::cout << "**** Tiles construction time: " << mytimer.elapsed_msec() << " ms." << std::endl;
		mytimer.start();
		for(unsigned int i = 0; i < num_tiles_; ++ i)
			tiles_[i].init(loading[i], base_norm_, in_pattern_, vandermonde_mat_, mask_mat_);
		mytimer.stop();
		std::cout << "**** Tiles init time: " << mytimer.elapsed_msec() << " ms." << std::endl;

		return true;
	} // RMC::init()


	// simulate RMC
	bool RMC::simulate(int num_steps, real_t tstar, unsigned int rate) {

		#ifdef USE_GPU
			tiles_[0].update_f_mats();
		#endif
		tiles_[0].save_mat_image(0);
		tiles_[0].save_mat_image_direct(0);
		unsigned int ten_percent = floor(num_steps / 10);
		unsigned int curr_percent = 10;
		std::cout << "++ performing simulation ..." << std::endl;
		for(unsigned int step = 0; step < num_steps; ++ step) {
			//std::cout << "." << std::flush;
			if((step + 1) % ten_percent == 0) {
				std::cout << curr_percent << "\% done at step " << step + 1 << std::endl;
				curr_percent += 10;
			} // if
			for(unsigned int i = 0; i < num_tiles_; ++ i) {
				tiles_[i].simulate_step(in_pattern_, vandermonde_mat_, mask_mat_, tstar, base_norm_);
				if((step + 1) % rate == 0) tiles_[i].update_model(in_pattern_, base_norm_);
				//if(step % 100 == 0) {
				//	tiles_[0].update_model(in_pattern_, base_norm_);
				//	tiles_[0].save_mat_image((step / 100 + 1));
				//} // if
			} // for
		} // for
		std::cout << std::endl << "++ simulation finished" << std::endl;
		for(unsigned int i = 0; i < num_tiles_; ++ i) {
			double chi2 = 0.0;
			mat_real_t a(size_, size_);
			tiles_[i].finalize_result(chi2, a);
			tiles_[i].print_times();
		} // for
		tiles_[0].save_mat_image(1);		// save mod_f mat
		tiles_[0].save_mat_image_direct(1);	// save a_mat
		tiles_[0].save_chi2_list();

		return true;
	} // RMC::simulate()


	bool RMC::scale(unsigned int final_size) {
		//tiles_[0].print_a_mat();

		unsigned int num_steps = final_size - size_;
		for(unsigned int i = 0; i < num_steps; ++ i) {
			// make sure all indices info is in a_mat_
			tiles_[0].update_model(in_pattern_, base_norm_);
			tiles_[0].scale_step(in_pattern_, base_norm_);
			//tiles_[0].print_a_mat();
			tiles_[0].save_mat_image_direct(i);
			// update indices_ ... and other stuff ... using the new model
			tiles_[0].update_from_model();
		} // for

		//tiles_[0].print_a_mat();

		return true;
	} // RMC::scale()



} // namespace hir

#endif // __RMC_HPP__
