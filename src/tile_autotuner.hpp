/**
 *  Project: HipRMC
 *
 *  File: tile_autotuner.hpp
 *  Created: Sep 05, 2013
 *  Modified: Sun 08 Sep 2013 10:03:05 AM PDT
 *
 *  Author: Abhinav Sarje <asarje@lbl.gov>
 */

#ifndef __TILE_AUTOTUNER_HPP__
#define __TILE_AUTOTUNER_HPP__

#include "typedefs.hpp"

namespace hir {

	class TileAutotuner {
		private:
			mat_real_t a_mat_;					// the model matrix
			unsigned int size_;
			vec_uint_t indices_;				// indices array
			unsigned int num_particles_;

			real_t tstar_;						// current temperature
			real_t cooling_factor_;				// cooling factor - keep it as 0 for now ...

			mat_complex_t f_mat_;				// stores fft
			mat_real_t mod_f_mat_;				// stores mod of fft
			mat_complex_t dft_mat_;				// dft matrix

			real_t prev_chi2_;					// the chi2 error of last accepted step

			unsigned int accepted_moves_;		// number of accepted moves

			bool init(const vec_uint_t&, unsigned int, real_t);
			bool update_a_mat();
			bool scale_step();

		public:
			TileAutotuner(unsigned int, unsigned int, const vec_uint_t&);
			TileAutotuner(const TileAutotuner&);
			~TileAutotuner();

		friend class Tile;
	}; // class TileAutotuner

} // namespace hir

#endif // __TILE_AUTOTUNER_HPP__
