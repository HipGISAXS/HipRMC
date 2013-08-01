/***
  *  Project: HipRMC - High Performance Reverse Monte Carlo Simulations
  *
  *  File: rmc.cpp
  *  Created: Jan 25, 2013
  *  Modified: Thu 01 Aug 2013 12:16:03 PM PDT
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#include <random>
#include <algorithm>
#include <boost/filesystem.hpp>

#include "rmc.hpp"
#include "constants.hpp"
#include "hiprmc_input.hpp"
#ifdef USE_GPU
#include "init_gpu.cuh"
#endif

namespace hir {

	RMC::RMC(char* filename) :
			in_pattern_(0, 0),
			scaled_pattern_(0, 0),
			mask_mat_(0, 0),
			vandermonde_mat_(0, 0) {
		if(!HipRMCInput::instance().construct_input_config(filename)) {
			std::cerr << "error: failed to construct input configuration" << std::endl;
			exit(1);
		} // if

		HipRMCInput::instance().print_all();

		rows_ = HipRMCInput::instance().num_rows();
		cols_ = HipRMCInput::instance().num_cols();
		size_ = std::max(rows_, cols_);
		num_tiles_ = HipRMCInput::instance().num_tiles();
		in_pattern_.resize(rows_, cols_);
		unsigned int start_num_rows = HipRMCInput::instance().model_start_num_rows();
		unsigned int start_num_cols = HipRMCInput::instance().model_start_num_cols();
		tile_size_ = std::max(start_num_rows, start_num_cols);
		scaled_pattern_.resize(tile_size_, tile_size_);
		mask_mat_.resize(tile_size_, tile_size_);
		vandermonde_mat_.resize(tile_size_, tile_size_);

		// for now only square patterns are considered
		if(rows_ != cols_) {
			std::cerr << "error: number of rows should equal number of columns" << std::endl;
			exit(1);
		} // if
		if(tile_size_ > rows_) {
			std::cerr << "error: initial tile size should be less or equal to pattern size" << std::endl;
			exit(1);
		} // if
		#ifdef USE_GPU
			if(!init_gpu()) {
				std::cerr << "error: " << std::endl;
				exit(1);
			} // if
		#endif
		#ifdef USE_MPI
			if(!init_mpi(1, &filename)) {
				std::cerr << "error: " << std::endl;
				exit(1);
			} // if
		#endif
		if(!init()) {
			std::cerr << "error: failed to pre-initialize RMC object" << std::endl;
			exit(1);
		} // if
	} // RMC::RMC()


	RMC::RMC(int narg, char** args, unsigned int rows, unsigned int cols, const char* img_file,
					unsigned int num_tiles, unsigned int init_tile_size, real_t* loading) :
			in_pattern_(rows, cols),
			rows_(rows), cols_(cols), size_(std::max(rows, cols)),
			//in_mask_(NULL),
			//in_mask_len_(0),
			num_tiles_(num_tiles),
			tile_size_(init_tile_size),
			scaled_pattern_(init_tile_size, init_tile_size),
			mask_mat_(init_tile_size, init_tile_size),
			vandermonde_mat_(init_tile_size, init_tile_size) {
		// for now only square patterns are considered
		if(rows_ != cols_) {
			std::cerr << "error: number of rows should equal number of columns" << std::endl;
			exit(1);
		} // if
		if(tile_size_ > rows_) {
			std::cerr << "error: initial tile size should be less or equal to pattern size" << std::endl;
			exit(1);
		} // if
		if(!init(narg, args, img_file, loading)) {
			std::cerr << "error: failed to pre-initialize RMC object" << std::endl;
			exit(1);
		} // if
	} // RMC::RMC()


	RMC::~RMC() {
		//if(in_mask_ != NULL) delete[] in_mask_;
		//if(mask_mat_ != NULL) delete[] mask_mat_;
	} // RMC::~RMC()


	// idea is that this can be replaced easily for other types of raw inputs (non image)
	bool RMC::init(int narg, char** args, const char* img_file, real_t* loading) {
		std::cout << "++ init" << std::endl;
		#ifdef USE_GPU
			if(!init_gpu()) {
				std::cerr << "error: " << std::endl;
				return false;
			} // if
		#endif
		#ifdef USE_MPI
			if(!init_mpi(narg, args)) {
				std::cerr << "error: " << std::endl;
				return false;
			} // if
		#endif

		#ifdef USE_MPI
		if(main_comm.rank() == 0) {
		#endif
			// TODO: opencv usage is temporary. improve with something else...
			// TODO: take a subimage of the input ...
			cv::Mat img = cv::imread(img_file, 0);	// grayscale only for now
			//cv::getRectSubPix(img, cv::Size(rows_, cols_), cv::Point2f(cx, cy), subimg);
			// extract the input image raw data (grayscale values)
			// and create mask array = indices in image data where value is min
			real_t *img_data = new (std::nothrow) real_t[rows_ * cols_];
			//unsigned int *mask_data = new (std::nothrow) unsigned int[rows_ * cols_];
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
			// initialize image data from img
			for(unsigned int i = 0; i < rows_; ++ i) {
				for(unsigned int j = 0; j < cols_; ++ j) {
					unsigned int temp = (unsigned int) img.at<unsigned char>(i, j);
					
					// do the quadrant swap thingy ...
					//unsigned int img_index = cols_ * ((i + hrow) % rows_) + (j + hcol) % cols_;
					// or not ...
					unsigned int img_index = cols_ * i + j;
					img_data[img_index] = (real_t) temp;
					//if(temp == 0) mask_data[mask_count ++] = img_index;
				} // for
			} // for
			for(unsigned int i = 0; i < rows_; ++ i) {
				for(unsigned int j = 0; j < cols_; ++ j) {
					img.at<unsigned char>(i, j) = (unsigned char) img_data[cols_ * i + j];
				} // for
			} // for
			// write it out
			cv::imwrite("base_pattern.tif", img);

		#ifdef USE_MPI
			// TODO: send img_data to all procs ...
		} else {
			// TODO: receive img_data from proc 0 ...
		} // if-else
		#endif

		// TODO: for now, limit to max num procs == num tiles ...

		//print_matrix("img_data:", img_data, rows_, cols_);
		//print_array("mask_data:", mask_data, mask_count);

		in_pattern_.populate(img_data);
		vec_uint_t indices;
		initialize_particles_random(indices);

		initialize_simulation(1);
		initialize_tiles(indices, loading);

		//delete[] mask_data;
		delete[] img_data;
		return true;
	} // RMC::init()


	bool RMC::init() {
		std::cout << "++ init" << std::endl;

		#ifdef USE_MPI
		if(main_comm.rank() == 0) {
		#endif
			// create output directory first
			const std::string p = HipRMCInput::instance().label();
			if(!boost::filesystem::create_directory(p)) {
				std::cerr << "error: could not create output directory " << p << std::endl;
				return false;
			} // if

			// TODO: opencv usage is temporary. improve with something else...
			// TODO: take a subimage of the input ...
			cv::Mat img = cv::imread(HipRMCInput::instance().input_image(), 0);	// grayscale only for now
			//cv::getRectSubPix(img, cv::Size(rows_, cols_), cv::Point2f(cx, cy), subimg);
			// extract the input image raw data (grayscale values)
			// and create mask array = indices in image data where value is min
			real_t *img_data = new (std::nothrow) real_t[rows_ * cols_];
			//unsigned int *mask_data = new (std::nothrow) unsigned int[rows_ * cols_];
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
			// initialize image data from img
			for(unsigned int i = 0; i < rows_; ++ i) {
				for(unsigned int j = 0; j < cols_; ++ j) {
					unsigned int temp = (unsigned int) img.at<unsigned char>(i, j);
					
					// do the quadrant swap thingy ...
					//unsigned int img_index = cols_ * ((i + hrow) % rows_) + (j + hcol) % cols_;
					// or not ...
					unsigned int img_index = cols_ * i + j;
					img_data[img_index] = (real_t) temp;
					//if(temp == 0) mask_data[mask_count ++] = img_index;
				} // for
			} // for
			for(unsigned int i = 0; i < rows_; ++ i) {
				for(unsigned int j = 0; j < cols_; ++ j) {
					img.at<unsigned char>(i, j) = (unsigned char) img_data[cols_ * i + j];
				} // for
			} // for
			// write it out
			cv::imwrite(HipRMCInput::instance().label() + "/base_pattern.tif", img);

		#ifdef USE_MPI
			// TODO: send img_data to all procs ...
		} else {
			// TODO: receive img_data from proc 0 ...
		} // if-else
		#endif

		// TODO: for now, limit to max num procs == num tiles ...

		//print_matrix("img_data:", img_data, rows_, cols_);
		//print_array("mask_data:", mask_data, mask_count);

		in_pattern_.populate(img_data);
		vec_uint_t indices;
		initialize_particles_random(indices);

		initialize_simulation(1);
		initialize_tiles(indices, &(HipRMCInput::instance().loading()[0]));

		//delete[] mask_data;
		delete[] img_data;
		return true;
	} // RMC::init()


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


	// this is for every simulation set
	bool RMC::initialize_simulation(unsigned int scale_factor) {
		// scale pattern to current size
		scale_pattern_to_tile(scale_factor);
		// process pattern, scale pixel intensities
		preprocess_pattern_and_mask(scale_factor);
		compute_base_norm();
		initialize_vandermonde(scale_factor);

		return true;
	} // RMC::initialize_simulation()


	bool RMC::initialize_simulation_tiles() {
		for(unsigned int i = 0; i < num_tiles_; ++ i) {
			tiles_[i].init_scale(base_norm_, scaled_pattern_, vandermonde_mat_, mask_mat_);
		} // for
		return true;
	} // RMC::initialize_simulation_tiles()


	bool RMC::initialize_tiles(const vec_uint_t &indices, const real_t* loading) {
		std::cout << "++ initialize_tiles " << num_tiles_ << std::endl;
		// initialize tiles
		for(unsigned int i = 0; i < num_tiles_; ++ i)
			tiles_.push_back(Tile(tile_size_, tile_size_, indices, size_));
		for(unsigned int i = 0; i < num_tiles_; ++ i)
			tiles_[i].init(loading[i], base_norm_, scaled_pattern_, vandermonde_mat_, mask_mat_);
		return true;
	} // RMC::initialize_tiles()


	// check ...
	bool RMC::initialize_vandermonde(unsigned int scale_fac) {
		// compute vandermonde matrix
		// generate 1st order power (full 360 deg rotation in polar coords)
		std::vector<complex_t> first_pow;
		for(unsigned int i = 0; i < tile_size_; ++ i) {
			real_t temp = 2.0 * PI_ * (1.0 - ((real_t) i / tile_size_));
			real_t temp_r = cos(temp);
			real_t temp_i = sin(temp);
			first_pow.push_back(complex_t(temp_r, temp_i));
		} // for
		//print_carray("first_pow", reinterpret_cast<complex_t*>(&first_pow[0]), size_);
		if(vandermonde_mat_.num_rows() + scale_fac == tile_size_) {
			vandermonde_mat_.incr_rows(scale_fac);
			vandermonde_mat_.incr_columns(scale_fac);
		} else if(vandermonde_mat_.num_rows() != tile_size_) {
			std::cerr << "error: Mr. Vandermonde is very angry! "
						<< vandermonde_mat_.num_rows() << ", " << tile_size_ << std::endl;
			return false;
		} // if-else
		// initialize first column
		typename mat_complex_t::col_iterator citer = vandermonde_mat_.column(0);
		for(unsigned int i = 0; i < citer.size(); ++ i) citer[i] = complex_t(1.0, 0.0);
		// compute rest of the matrix
		typename mat_complex_t::col_iterator curr_citer = vandermonde_mat_.begin_col();
		typename mat_complex_t::col_iterator prev_citer = vandermonde_mat_.begin_col();
		++ curr_citer;
		for(; curr_citer != vandermonde_mat_.end_col(); ++ curr_citer, ++ prev_citer) {
			for(unsigned int i = 0; i < tile_size_; ++ i) curr_citer[i] = prev_citer[i] * first_pow[i];
		} // while
		//print_cmatrix("vandermonde_mat", vandermonde_mat_.data(), size_, size_);

		return true;
	} // RMC::initialize_vandermonde()


	bool RMC::initialize_particles_random(vec_uint_t &indices) {
		indices.clear();
		// create array of random indices
		for(unsigned int i = 0; i < tile_size_ * tile_size_; ++ i) indices.push_back(i);
		// using mersenne-twister
		std::random_device rd;
		std::mt19937_64 gen(rd());
		std::shuffle(indices.begin(), indices.end(), gen);
		//print_array("indices", (unsigned int*)&indices[0], indices.size());
		return true;
	} // RMC::initialize_particles_random()


	bool RMC::scale_pattern_to_tile(unsigned int scale_factor) {
		if(size_ == tile_size_) {
			scaled_pattern_ = in_pattern_;
		} else {
			real_t* pdata = new (std::nothrow) real_t[size_ * size_];
			memcpy(pdata, in_pattern_.data(), size_ * size_ * sizeof(real_t));
			real_t* scaled_pdata = NULL;
			wil::scale_image((int) size_, (int) size_, (int) tile_size_, (int) tile_size_,
								pdata, scaled_pdata);
			// increase the size of the scaled pattern
			if(scaled_pattern_.num_rows() + scale_factor == tile_size_) {
				scaled_pattern_.incr_rows(scale_factor);
				scaled_pattern_.incr_columns(scale_factor);
			} // if
			// populate with the scaled data
			scaled_pattern_.populate(scaled_pdata);
			delete[] scaled_pdata;
		} // if-else
		return true;
	} // RMC::scale_pattern_to_tile()


	bool RMC::preprocess_pattern_and_mask(unsigned int scale_fac) {
		double min_val, max_val;
		woo::matrix_min_max(scaled_pattern_, min_val, max_val);
		double threshold = min_val;// + 2 * ceil(max_val / (min_val + 1));
		std::cout << "MIN: " << min_val << ", MAX: " << max_val << ", THRESH: " << threshold << std::endl;
		// sanity check
		if(scaled_pattern_.num_rows() != tile_size_) {
			std::cerr << "error: you are now really in grave danger: "
						<< scaled_pattern_.num_rows() << ", " << tile_size_ << std::endl;
			return false;
		} else {
			//std::cout << "be happiee: " << scaled_pattern_.num_rows() << ", " << tile_size_ << std::endl;
		} // if-else
		// apply threshold and
		// scale pixel intensities to span all of 0 - 255
		// and generate mask_mat_
		//memset(mask_mat_, 0, tile_size_ * tile_size_ * sizeof(unsigned int));
		if(mask_mat_.num_rows() + scale_fac == tile_size_) {
			mask_mat_.incr_rows(scale_fac);
			mask_mat_.incr_columns(scale_fac);
		} else if(mask_mat_.num_rows() != tile_size_) {
			std::cerr << "error: you have a wrong mask. "
						<< mask_mat_.num_rows() << ", " << tile_size_ << std::endl;
			return false;
		} // if-else
		mask_mat_.fill(1);
		for(unsigned int i = 0; i < tile_size_; ++ i) {
			for(unsigned int j = 0; j < tile_size_; ++ j) {
				double temp;
				if(scaled_pattern_(i, j) > threshold) {
					temp = 255 * (scaled_pattern_(i, j) - threshold) / (max_val - threshold);
				} else {
					temp = 0.0;
					mask_mat_(i, j) = 0;
				} // if-else
				scaled_pattern_(i, j) = temp;
			} // for
		} // for
		return true;
	} // RMC::preprocess_pattern_and_mask()


	bool RMC::compute_base_norm() {
		// compute base norm
		base_norm_ = 0.0;				// why till size/2 only ??? and what is Y ???
		unsigned int maxi = tile_size_;		// >> 1;
		for(unsigned int i = 0; i < maxi; ++ i) {
			for(unsigned int j = 0; j < maxi; ++ j) {
				base_norm_ += scaled_pattern_(i, j); // * (j + 1);	// skipping creation of Y matrix
			} // for
		} // for
		std::cout << "++ base_norm: " << base_norm_ << std::endl;
		return true;
	} // RMC::compute_base_norm();


	/*bool RMC::initialize_mask() {
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

		return true;
	} // RMC::initialize_mask()*/


	// simulate RMC
	bool RMC::simulate(int num_steps, real_t tstar, unsigned int rate, unsigned int scale_factor = 1) {

		if(!initialize_simulation(scale_factor)) {
			std::cerr << "error: failed to initialize simulation set" << std::endl;
			return false;
		} // if
		if(!initialize_simulation_tiles()) {
			std::cerr << "error: failed to initialize simulation set" << std::endl;
			return false;
		} // if

		std::cout << "Tile size: " << tile_size_ << std::endl;

		#ifdef USE_GPU
			tiles_[0].update_f_mats();
		#endif
		for(unsigned int i = 0; i < num_tiles_; ++ i) {
			std::cout << "saving initial images ..." << std::endl;
			tiles_[i].save_mat_image(i);
			tiles_[i].save_fmat_image(i);
			tiles_[i].save_mat_image_direct(i);
		} // for
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
				tiles_[i].simulate_step(scaled_pattern_, vandermonde_mat_, mask_mat_, tstar, base_norm_);
				if((step + 1) % rate == 0) tiles_[i].update_model();
				/*if(step % 100 == 0) {
					tiles_[i].update_model();
					#ifdef USE_GPU
						tiles_[i].update_f_mats();
					#endif
					tiles_[i].save_mat_image((step / 100 + 1));
					tiles_[i].save_mat_image_direct(step / 100 + 1);	// save a_mat
				} // if*/
			} // for
		} // for
		std::cout << std::endl << "++ simulation finished" << std::endl;
		for(unsigned int i = 0; i < num_tiles_; ++ i) {
			double chi2 = 0.0;
			mat_real_t a(tile_size_, tile_size_);
			tiles_[i].finalize_result(chi2, a);
			std::cout << "++++ final chi2 = " << chi2 << std::endl;
			tiles_[i].print_times();

			#ifdef USE_GPU
			// temp ... for bandwidth computation of dft2 ...
			unsigned int num_blocks = ceil((real_t) tile_size_ / CUDA_BLOCK_SIZE_X_) *
										ceil((real_t) tile_size_ / CUDA_BLOCK_SIZE_Y_);
			unsigned int read_bytes = num_blocks * (CUDA_BLOCK_SIZE_X_ + CUDA_BLOCK_SIZE_Y_) *
										sizeof(cucomplex_t);
			unsigned int write_bytes = num_blocks * CUDA_BLOCK_SIZE_X_ * CUDA_BLOCK_SIZE_Y_ *
										sizeof(cucomplex_t);
			std::cout << "+++++++ DFT2: "
					<< (float) (read_bytes + write_bytes) * num_steps * 1000 /
						(tiles_[i].dft2_time * 1024 * 1024)
					<< " MB/s" << std::endl;
			#endif

			std::cout << "saving images ... " << std::endl;
			tiles_[i].save_mat_image(num_tiles_ + i);		// save mod_f mat
			tiles_[i].save_fmat_image(num_tiles_ + i);		// save mod_f mat
			tiles_[i].save_mat_image_direct(num_tiles_ + i);	// save a_mat*/
			tiles_[i].save_chi2_list(i);
		} // for

		return true;
	} // RMC::simulate()


	bool RMC::simulate_and_scale(int num_steps_fac, unsigned int scale_factor,
								real_t tstar, unsigned int rate) {
		std::cout << "++ performing scaling and simulation ..." << std::endl;
		unsigned int num_steps = num_steps_fac * tile_size_;
		unsigned int curr_scale_fac = scale_factor;
		simulate(num_steps, tstar, rate, scale_factor);
		for(unsigned int tsize = tile_size_, iter = 0; tsize < size_; tsize += curr_scale_fac, ++ iter) {
			if(tile_size_ < size_) {
				for(unsigned int i = 0; i < num_tiles_; ++ i) {
					tiles_[i].update_model();
					if(tile_size_ + scale_factor > size_) {
						curr_scale_fac = size_ - tile_size_;
					} // if
					for(unsigned int s = 0; s < curr_scale_fac; ++ s) {
						tiles_[i].scale_step();
					} // for
					if(tiles_[i].size() != tile_size_ + curr_scale_fac) {
						std::cerr << "error: you are in graaaaaaave danger!" << std::endl;
						return false;
					} // if
					//tiles_[i].save_mat_image_direct(i);
				} // for
				tile_size_ += curr_scale_fac;
			} // if
			num_steps = num_steps_fac * tile_size_;
			simulate(num_steps, tstar, rate, curr_scale_fac);
		} // for
		return true;
	} // RMC::simulate_and_scale()


	bool RMC::simulate_and_scale() {
		std::cout << "++ performing scaling and simulation ..." << std::endl;
		int num_steps_fac = HipRMCInput::instance().num_steps_factor();
		unsigned int scale_factor = HipRMCInput::instance().scale_factor();
		real_t tstar = 1;				// FIXME ... hardcoded ...
		unsigned int rate = 10000;		// FIXME ... hardcoded ...
		unsigned int num_steps = num_steps_fac * tile_size_;
		unsigned int curr_scale_fac = scale_factor;
		simulate(num_steps, tstar, rate, scale_factor);
		for(unsigned int tsize = tile_size_, iter = 0; tsize < size_; tsize += curr_scale_fac, ++ iter) {
			if(tile_size_ < size_) {
				for(unsigned int i = 0; i < num_tiles_; ++ i) {
					tiles_[i].update_model();
					if(tile_size_ + scale_factor > size_) {
						curr_scale_fac = size_ - tile_size_;
					} // if
					for(unsigned int s = 0; s < curr_scale_fac; ++ s) {
						tiles_[i].scale_step();
					} // for
					if(tiles_[i].size() != tile_size_ + curr_scale_fac) {
						std::cerr << "error: you are in graaaaaaave danger!" << std::endl;
						return false;
					} // if
					//tiles_[i].save_mat_image_direct(i);
				} // for
				tile_size_ += curr_scale_fac;
			} // if
			num_steps = num_steps_fac * tile_size_;
			simulate(num_steps, tstar, rate, curr_scale_fac);
		} // for
		return true;
	} // RMC::simulate_and_scale()


	// this is for testing scaling
	// it scaled in_pattern_ itself
	bool RMC::scale(unsigned int final_size) {
		//tiles_[0].print_a_mat();

		unsigned int num_steps = final_size - size_;
		for(unsigned int i = 0; i < num_steps; ++ i) {
			// make sure all indices info is in a_mat_
			tiles_[0].update_model();
			tiles_[0].scale_step();
			//tiles_[0].print_a_mat();
			tiles_[0].save_mat_image_direct(i);
			// update indices_ ... and other stuff ... using the new model
			tiles_[0].update_from_model();
		} // for

		//tiles_[0].print_a_mat();

		return true;
	} // RMC::scale()

} // namespace hir
