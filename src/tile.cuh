/***
  *  Project:
  *
  *  File: tile.cuh
  *  Created: Feb 02, 2013
  *  Modified: Mon 04 Feb 2013 11:39:33 AM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef __TILE_CUH__
#define __TILE_CUH__

#include <cufft.h>

#include "typedefs.hpp"

namespace hir {

	class GTile {
		friend class Tile;

		private:
			dim3 block_dims_;
			dim3 grid_dims_;
			unsigned int size_;

			// device buffers
			cucomplex_t* a_mat_;
			cucomplex_t* f_mat_[2];
			real_t* mod_f_mat_[2];
			// host buffers
			cucomplex_t* complex_buff_h_;
			real_t* real_buff_h_;

		public:
			__host__ GTile();
			__host__ ~GTile();

			__host__ bool init(unsigned int, unsigned int, unsigned int);
			__host__ bool set_a_mat(real_t*);

			__host__ bool compute_fft_mat(unsigned int);
			__host__ cufftResult execute_cufft(cufftHandle, cuFloatComplex*, cuFloatComplex*);
			__host__ cufftResult execute_cufft(cufftHandle, cuDoubleComplex*, cuDoubleComplex*);
			__host__ bool normalize_fft_mat(unsigned int, unsigned int);
			__host__ bool compute_mod_mat(unsigned int, unsigned int);
	}; // class GTile

	__global__ void compute_mod_mat_kernel(cucomplex_t*, unsigned int, real_t*);
	__global__ void normalize_fft_mat_kernel(cucomplex_t*, unsigned int, unsigned int);

} // namespace hir

#endif // __TILE_CUH__
