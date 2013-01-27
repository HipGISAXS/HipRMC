/***
  *  Project: HipRMC - High Performance Reverse Monte Carlo Simulations
  *
  *  File: rmc.hpp
  *  Created: Jan 25, 2013
  *  Modified: Sun 27 Jan 2013 12:18:19 PM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef __RMC_HPP__
#define __RMC_HPP__

#include <opencv2/opencv.hpp>
#include <woo/matrix/matrix.hpp>

namespace hir {

	template <typename real_t>
	class RMC {
		private:
			//unsigned int in_rows_;	// number of rows in input pattern
			//unsigned int in_cols_;	// number of columns in input pattern
			//real_t* in_pattern_;		// the input pattern
			woo::Matrix2D<real_t> in_pattern_;	// input pattern and related matrix info
			// any benefit of using vectors instead? ...
			unsigned int* in_mask_;		// the input mask
			real_t* loading_factors_;	// array of loading factor of each tile
			unsigned int in_mask_len_;	// size of input mask
			unsigned int num_tiles_;	// total number of tiles

			bool pre_init();			// to extract data from image
			bool init();				// initialize with raw data

		public:
			RMC();
			~RMC();
			bool simulate(int num_steps);
	}; // class RMC


	template <typename real_t>
	RMC<real_t>::RMC() {
		if(!pre_init()) {
			std::cerr << "error: failed to pre-initialize RMC object" << std::endl;
			delete[] img_data;
			exit(1);
		} // if
	} // RMC::RMC()


	template <typename real_t>
	RMC<real_t>::~RMC() {
		if(in_mask_ != NULL) delete[] in_mask_;
		if(loading_factors_ != NULL) delete[] loading_factors_;
	} // RMC::~RMC()


	// idea is that this can be replaced easily for other types of raw inputs (not image)
	template <typename real_t>
	bool RMC<real_t>::pre_init(unsigned int rows, unsigned int cols, char* img_file,
								unsigned int num_tiles, real_t* loading) {
		// TODO: opencv usage is temporary. improve with something else...
		cv::Mat img = cv::imread(img_file, 0);	// grayscale only for now ...
		// extract the input image raw data (grayscale values)
		// and create mask array = indices in image data where value is 0
		real_t *img_data = new (std::nothrow) real_t[rows * cols];
		unsigned int *mask_data = new (std::nothrow) unsigned int[rows * cols];
		unsigned int mask_count = 0;
		for(unsigned int i = 0; i < rows; ++ i) {
			for(unsigned int j = 0; j < cols; ++ j) {
				unsigned int temp = (unsigned int) img.at<unsigned char>(i, j);
				img_data[cols * i + j] = (real_t) temp;
				if(temp == 0) mask_data[mask_count ++] = cols * i + j;
			} // for
		} // for

		if(!init(rows, cols, img_data, mask_count, mask_data, num_tiles, loading)) {
			std::cerr << "error: failed to initialize RMC object" << std::endl;
			delete[] img_data;
			return false;
		} // if

		delete[] img_data;
		return true;
	} // RMC::pre_init()


	// initialize with raw data
	template <typename real_t>
	bool RMC<real_t>::init(unsigned int rows, unsigned int cols, real_t* pattern,
							unsigned int mask_len, unsigned int* mask,
							unsigned int num_tiles, real_t* loading) :
			in_pattern_(rows, cols, pattern),
			in_mask_len_(mask_len),
			num_tiles_(num_tiles),
			in_mask_(NULL), loading_factors_(NULL) {
		// for now only square patterns are considered
		if(rows != cols) return false;

		// create mask and loading arays
		in_mask_ = new (std::nothrow) unsigned int[mask_len];
		loading_factors_ = new (std::nothrow) real_t[num_tiles];
		if(in_mask_ == NULL || loading_factors_ == NULL) return false;
		memcpy(in_mask_, mask, mask_len * sizeof(unsigned int));
		memcpy(loading_factors_, loading, num_tiles * sizeof(real_t));

		return true;
	} // RMC::init()


	// simulate RMC
	template <typename real_t>
	bool RMC<real_t>::simulate(int num_steps) {

		return false;
	} // RMC::simulate()


} // namespace hir

#endif // __RMC_HPP__
