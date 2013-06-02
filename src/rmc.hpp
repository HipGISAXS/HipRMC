/***
  *  Project: HipRMC - High Performance Reverse Monte Carlo Simulations
  *
  *  File: rmc.hpp
  *  Created: Jan 25, 2013
  *  Modified: Mon 18 Mar 2013 05:53:45 PM PDT
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef __RMC_HPP__
#define __RMC_HPP__

#include <opencv2/opencv.hpp>

#include "typedefs.hpp"
#include "tile.hpp"

namespace hir {

	class RMC {
		private:
			mat_real_t in_pattern_;	// input pattern and related matrix info
			unsigned int rows_;		// input pattern sizes
			unsigned int cols_;
			unsigned int size_;

			mat_real_t scaled_pattern_;	// current pattern, scaled to tile size
			unsigned int tile_size_;	// current tile size
										// any benefit of using vectors for below instead? ...
			//unsigned int* in_mask_;		// the input mask
			//unsigned int in_mask_len_;	// size of input mask
			//unsigned int* mask_mat_;	// mask matrix of 1 and 0
			mat_uint_t mask_mat_;		// mask matrix of 1 and 0

			unsigned int num_tiles_;	// total number of tiles
			vec_tile_t tiles_;			// the tiles
			mat_complex_t vandermonde_mat_;
			real_t base_norm_;			// norm of input

			#ifdef USE_MPI
			#endif

			// extracts raw data from image
			bool init(int, char**, const char*, real_t*);
			// initializes with raw data
			//bool init(real_t*, unsigned int, unsigned int*, real_t*);
			// initialization for each set of simulation runs
			bool initialize_tiles(const vec_uint_t&, const real_t*);
			bool initialize_vandermonde(unsigned int);
			bool initialize_particles_random(vec_uint_t&);
			bool initialize_simulation(unsigned int);
			bool initialize_simulation_tiles();
			bool compute_base_norm();
			bool initialize_mask();
			bool scale_image_colormap(cv::Mat&, double, double);
			bool scale_pattern_to_tile(unsigned int);
			bool preprocess_pattern_and_mask(unsigned int);

		public:
			RMC(int, char**, unsigned int, unsigned int, const char*, unsigned int, unsigned int, real_t*);
			~RMC();
			bool simulate(int, real_t, unsigned int, unsigned int);
			bool simulate_and_scale(int, unsigned int, real_t, unsigned int);

			// for testing ...
			bool scale(unsigned int size);
	}; // class RMC

} // namespace hir

#endif // __RMC_HPP__
