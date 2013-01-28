/***
  *  Project: HipRMC - High Performance Reverse Monte Carlo Simulations
  *
  *  File: rmc.hpp
  *  Created: Jan 25, 2013
  *  Modified: Sun 27 Jan 2013 07:05:25 PM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef __RMC_HPP__
#define __RMC_HPP__

#include <opencv2/opencv.hpp>
#include <woo/matrix/matrix.hpp>

#include "tile.hpp"

namespace hir {

	template <typename real_t>
	class RMC {
		private:
			woo::Matrix2D<real_t> in_pattern_;	// input pattern and related matrix info
										// any benefit of using vectors for below instead? ...
			unsigned int* in_mask_;		// the input mask
			unsigned int in_mask_len_;	// size of input mask
			unsigned int num_tiles_;	// total number of tiles
			Tile* tiles_;				// the tiles -- temp

			// extract raw data from image
			bool pre_init(unsigned int, unsigned int, char*, unsigned int, real_t*);
			// initialize with raw data
			bool init(unsigned int, unsigned int, real_t*, unsigned int, unsigned int*,
					unsigned int, real_t*);

		public:
			RMC(unsigned int, unsigned int, char*, unsigned int, real_t*);
			~RMC();
			bool simulate(int num_steps);
	}; // class RMC


	template <typename real_t>
	RMC<real_t>::RMC(unsigned int rows, unsigned int cols, char* img_file,
					unsigned int num_tiles, real_t* loading) :
		in_pattern_(rows, cols),
		in_mask_(NULL),
		in_mask_len_(0),
		num_tiles_(num_tiles),
		tiles_(NULL) {

		if(!pre_init(rows, cols, img_file, num_tiles, loading)) {
			std::cerr << "error: failed to pre-initialize RMC object" << std::endl;
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

		// TODO: take a subimage of the input

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
							unsigned int num_tiles, real_t* loading) {
		// for now only square patterns are considered
		if(rows != cols) return false;

		in_pattern_.populate(pattern);
		// create mask and loading arays
		in_mask_len_ = mask_len;
		in_mask_ = new (std::nothrow) unsigned int[mask_len];
		if(in_mask_ == NULL) return false;
		memcpy(in_mask_, mask, mask_len * sizeof(unsigned int));

		// initialize tiles

		unsigned int size = std::max(rows, cols);
		unsigned int size2 = size * size;
		// create array of random indices
		std::vector<unsigned int> indices;
		for(unsigned int i = 0; i < size2; ++ i) indices.push_back(i);
		std::random_shuffle(indices.begin(), indices.end());
		tiles_ = new (std::nothrow) Tile[num_tiles](size, size, indices);
		for(unsigned int i = 0; i < num_tiles; ++ i) tiles_[i].init(loading[i]);

		return true;
	} // RMC::init()


	// simulate RMC
	template <typename real_t>
	bool RMC<real_t>::simulate(int num_steps) {

		unsigned int size = std::max(rows, cols);
		unsigned int size2 = size * size;

		// compute base norm
		real_t base_norm = 0.0;		// why till size/2 ???
		for(unsigned int i = 0; i < size / 2; ++ i) {
			for(unsigned int j = 0; j < size / 2; ++ j) {
				base_norm += in_pattern_(i, j) * (j + 1);
			} // for
		} // for

		// generate mask matrix
		// TODO ...

		return false;
	} // RMC::simulate()


} // namespace hir

#endif // __RMC_HPP__
