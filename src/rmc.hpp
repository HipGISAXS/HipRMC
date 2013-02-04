/***
  *  Project: HipRMC - High Performance Reverse Monte Carlo Simulations
  *
  *  File: rmc.hpp
  *  Created: Jan 25, 2013
  *  Modified: Sun 03 Feb 2013 12:12:51 PM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef __RMC_HPP__
#define __RMC_HPP__

#include <random>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <woo/matrix/matrix.hpp>

#include "typedefs.hpp"
#include "tile.hpp"
#include "constants.hpp"

namespace hir {

//	template <typename real_t, typename complex_t, typename cucomplex_t>
	class RMC {
		private:
			woo::Matrix2D<real_t> in_pattern_;	// input pattern and related matrix info
			unsigned int rows_;
			unsigned int cols_;
			unsigned int size_;
										// any benefit of using vectors for below instead? ...
			unsigned int* in_mask_;		// the input mask
			unsigned int in_mask_len_;	// size of input mask
			unsigned int* mask_mat_;	// mask matrix of 1 and 0
			unsigned int num_tiles_;	// total number of tiles
//			std::vector<Tile<real_t, complex_t, cucomplex_t> > tiles_;	// the tiles -- temp
			std::vector<Tile> tiles_;	// the tiles -- temp
			woo::Matrix2D<complex_t> vandermonde_mat_;
			real_t base_norm_;			// norm of input

			// extracts raw data from image
			bool pre_init(const char*, real_t*);
			// initializes with raw data
			bool init(real_t*, unsigned int, unsigned int*, real_t*);

		public:
			RMC(unsigned int, unsigned int, const char*, unsigned int, real_t*);
			~RMC();
			bool simulate(int, real_t, unsigned int);
	}; // class RMC


//	template <typename real_t, typename complex_t, typename cucomplex_t>
	RMC::RMC(unsigned int rows, unsigned int cols, const char* img_file,
//	RMC<real_t, complex_t, cucomplex_t>::RMC(unsigned int rows, unsigned int cols, const char* img_file,
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


//	template <typename real_t, typename complex_t, typename cucomplex_t>
	RMC::~RMC() {
//	RMC<real_t, complex_t, cucomplex_t>::~RMC() {
		if(in_mask_ != NULL) delete[] in_mask_;
		if(mask_mat_ != NULL) delete[] mask_mat_;
		//if(tiles_ != NULL) delete[] tiles_;
	} // RMC::~RMC()


	// idea is that this can be replaced easily for other types of raw inputs (not image)
//	template <typename real_t, typename complex_t, typename cucomplex_t>
	bool RMC::pre_init(const char* img_file, real_t* loading) {
//	bool RMC<real_t, complex_t, cucomplex_t>::pre_init(const char* img_file, real_t* loading) {
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
		for(unsigned int i = 0; i < rows_; ++ i) {
			for(unsigned int j = 0; j < cols_; ++ j) {
				unsigned int temp = (unsigned int) img.at<unsigned char>(i, j);
				unsigned int img_index = cols_ * ((i + hrow) % rows_) + (j + hcol) % cols_;
				img_data[img_index] = (real_t) temp;
				if(temp == 0) mask_data[mask_count ++] = img_index;
			} // for
		} // for

		//print_matrix("img_data:", img_data, rows_, cols_);
		//print_array( "mask_data:", mask_data, mask_count);

		// TODO: take a subimage of the input

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


	// initialize with raw data
//	template <typename real_t, typename complex_t, typename cucomplex_t>
	bool RMC::init(real_t* pattern,
//	bool RMC<real_t, complex_t, cucomplex_t>::init(real_t* pattern,
													unsigned int mask_len, unsigned int* mask,
													real_t* loading) {
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
		base_norm_ = 0.0;		// why till size/2 only ???
		for(unsigned int i = 0; i < size_ / 2; ++ i) {
			for(unsigned int j = 0; j < size_ / 2; ++ j) {
				base_norm_ += in_pattern_(i, j) * (j + 1);	// skipping creation of Y matrix
			} // for
		} // for
		mytimer.stop();
		std::cout << "**** Base norm time: " << mytimer.elapsed_msec() << " ms." << std::endl;

		//std::cout << "++ base_norm: " << base_norm_ << std::endl;

		// initialize tiles

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

		mytimer.start();
		for(unsigned int i = 0; i < num_tiles_; ++ i)
			tiles_.push_back(Tile(size_, size_, indices));
			//tiles_.push_back(Tile<real_t, complex_t, cucomplex_t>(size_, size_, indices));
		mytimer.stop();
		std::cout << "**** Tiles construction time: " << mytimer.elapsed_msec() << " ms." << std::endl;
		mytimer.start();
		for(unsigned int i = 0; i < num_tiles_; ++ i)
			tiles_[i].init(loading[i], base_norm_, in_pattern_, mask_mat_);
		mytimer.stop();
		std::cout << "**** Tiles init time: " << mytimer.elapsed_msec() << " ms." << std::endl;

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
		typename woo::Matrix2D<complex_t>::col_iterator citer = vandermonde_mat_.column(0);
		for(unsigned int i = 0; i < citer.size(); ++ i) citer[i] = complex_t(1.0, 0.0);
		// compute rest of the matrix
		typename woo::Matrix2D<complex_t>::col_iterator curr_citer = vandermonde_mat_.begin_col();
		typename woo::Matrix2D<complex_t>::col_iterator prev_citer = vandermonde_mat_.begin_col();
		++ curr_citer;
		for(; curr_citer != vandermonde_mat_.end_col(); ++ curr_citer, ++ prev_citer) {
			for(unsigned int i = 0; i < size_; ++ i) curr_citer[i] = prev_citer[i] * first_pow[i];
		} // while
		mytimer.stop();
		std::cout << "**** Vandermonde time: " << mytimer.elapsed_msec() << " ms." << std::endl;
		//print_cmatrix("vandermonde_mat", vandermonde_mat_.data(), size_, size_);

		return true;
	} // RMC::init()


	// simulate RMC
//	template <typename real_t, typename complex_t, typename cucomplex_t>
	bool RMC::simulate(int num_steps, real_t tstar, unsigned int rate) {
//	bool RMC<real_t, complex_t, cucomplex_t>::simulate(int num_steps, real_t tstar, unsigned int rate) {

		std::cout << "++ starting simulation ..." << std::endl;
		for(unsigned int step = 0; step < num_steps; ++ step) {
			//std::cout << "+++ simulation step " << step << "..." << std::endl;
			std::cout << "." << std::flush;
			for(unsigned int i = 0; i < num_tiles_; ++ i) {
				tiles_[i].simulate_step(in_pattern_, vandermonde_mat_, mask_mat_, tstar, base_norm_);
				if((step + 1) % rate == 0) tiles_[i].update_model(in_pattern_, base_norm_);
			} // for
		} // for
		std::cout << std::endl << "++ simulation finished" << std::endl;
		for(unsigned int i = 0; i < num_tiles_; ++ i) {
			double chi2 = 0.0;
			woo::Matrix2D<real_t> a(size_, size_);
			tiles_[i].finalize_result(chi2, a);
		} // for

		return true;
	} // RMC::simulate()



} // namespace hir

#endif // __RMC_HPP__
