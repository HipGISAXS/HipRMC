/***
  *  Project: HipRMC - High Performance Reverse Monte Carlo Simulations
  *
  *  File: rmc.hpp
  *  Created: Jan 25, 2013
  *  Modified: Tue 04 Nov 2014 04:30:58 PM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef __RMC_HPP__
#define __RMC_HPP__

//#include <opencv2/opencv.hpp>

#ifdef USE_MPI
#include "woo/comm/multi_node_comm.hpp"
#endif
#include "typedefs.hpp"
#include "constants.hpp"
#include "tile.hpp"

namespace hir {

	class RMC {
		private:
			unsigned int rows_;					// input pattern sizes
			unsigned int cols_;
			unsigned int size_;

			// used for multi processors/MPI
			unsigned int local_rows_;			// local rows
			unsigned int local_cols_;			// local cols
			unsigned int matrix_offset_;		// offset into a matrix (num of elements)
			int row_offsets_[MAX_NUM_PROCS];	// row number offsets

			mat_real_t in_pattern_;				// local input pattern and related matrix info
			mat_uint_t mask_mat_;				// local mask matrix of 1 and 0

			//mat_real_t scaled_pattern_;		// current pattern, scaled to tile size
			mat_real_t cropped_pattern_;		// current local cropped pattern (current tile size)
			mat_uint_t cropped_mask_mat_;		// current local mask, crappoed to tile size
			mat_complex_t vandermonde_mat_;		// current local vandermonde matrix

			unsigned int tile_size_;			// current tile size
			unsigned int local_tile_rows_;		// current local tile num rows
			unsigned int local_tile_cols_;		// current local tile num cols
			unsigned int tile_offset_;			// offset into a current tile (num of elements)
			unsigned int tile_offset_rows_;		// rows offset into a current tile (num of rows)
			unsigned int tile_offset_cols_;		// cols offset into a current tile (num of cols)

			unsigned int global_num_tiles_;		// total number of tiles globally
			unsigned int num_tiles_;			// total number of lcoal tiles
			vec_tile_t tiles_;					// the local tiles

			real_t base_norm_;					// current norm of pattern

			#ifdef USE_MPI
				woo::MultiNode multi_node_;		// for communication across multiple nodes
			#endif

			// extracts raw data from image
			bool init();
			//bool init(int, char**, const char*, real_t*);
			// initializes with raw data
			//bool init(real_t*, unsigned int, unsigned int*, real_t*);
			// initialization for each set of simulation runs
			//bool initialize_tiles(const vec_uint_t&,
			//						const real_t*, const real_t*, const real_t*, unsigned int);
			bool initialize_tiles(const vec_uint_t&, const real_t*, unsigned int);
			bool initialize_vandermonde(unsigned int);
			bool initialize_particles_random(vec_uint_t&);
			bool initialize_particles_image(vec_uint_t&);
			bool initialize_simulation(unsigned int);
			bool initialize_simulation_tiles(int);
			bool destroy_simulation_tiles();
			bool compute_base_norm();
			bool initialize_mask();
//			bool scale_image_colormap(cv::Mat&, double, double);
			//bool scale_pattern_to_tile(unsigned int);
			bool crop_pattern_to_tile(unsigned int);
			bool preprocess_pattern_and_mask(unsigned int);

			bool normalize_cropped_pattern();

		public:
			RMC(int, char**, char*);
			~RMC();
			bool simulate(int, unsigned int, unsigned int);
			//bool simulate_and_scale(int, unsigned int, unsigned int);
			bool simulate_and_scale();

			// for testing ...
			bool scale(unsigned int size);
	}; // class RMC

} // namespace hir

#endif // __RMC_HPP__
