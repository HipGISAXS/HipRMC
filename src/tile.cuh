/***
  *  Project:
  *
  *  File: tile.cuh
  *  Created: Feb 02, 2013
  *  Modified: Mon 04 Feb 2013 03:12:06 PM PST
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
			real_t* pattern_;
			cucomplex_t* vandermonde_;
			cucomplex_t* a_mat_;
			unsigned int* mask_mat_;
			cucomplex_t* f_mat_[2];
			real_t* mod_f_mat_[2];
			cucomplex_t* dft_mat_;
			real_t* real_buff_d_;
			// host buffers
			cucomplex_t* complex_buff_h_;
			real_t* real_buff_h_;

		public:
			__host__ GTile();
			__host__ ~GTile();

			__host__ bool init(real_t*, cucomplex_t*, real_t*, const unsigned int*, unsigned int,
								unsigned int, unsigned int);

			__host__ bool compute_fft_mat(unsigned int);
			__host__ cufftResult execute_cufft(cufftHandle, cuFloatComplex*, cuFloatComplex*);
			__host__ cufftResult execute_cufft(cufftHandle, cuDoubleComplex*, cuDoubleComplex*);
			__host__ bool normalize_fft_mat(unsigned int, unsigned int);
			__host__ bool compute_mod_mat(unsigned int, unsigned int);
			__host__ bool mask_mat(unsigned int);
			__host__ bool copy_mod_mat(unsigned int);
			__host__ double compute_model_norm(unsigned int);
			__host__ double compute_chi2(unsigned int, real_t);
			__host__ bool compute_dft2(unsigned int, unsigned int, unsigned int, unsigned int, unsigned int);
			__host__ bool update_fft_mat(unsigned int, unsigned int);
	}; // class GTile

	// cuda kernels
	__global__ void compute_mod_mat_kernel(cucomplex_t*, unsigned int, real_t*);
	__global__ void normalize_fft_mat_kernel(cucomplex_t*, unsigned int, unsigned int);
	__global__ void mask_mat_kernel(unsigned int*, unsigned int, real_t*);
	__global__ void compute_model_norm_kernel(real_t*, unsigned int, unsigned int, real_t*);
	__global__ void compute_chi2_kernel(real_t*, real_t*, unsigned int, real_t, real_t*);
	__global__ void compute_dft2_kernel(cucomplex_t*, unsigned int, unsigned int, unsigned int,
										unsigned int, unsigned int, unsigned int, cucomplex_t*);
	__global__ void update_fft_mat_kernel(cucomplex_t*, cucomplex_t*, unsigned int, cucomplex_t*);

} // namespace hir

#endif // __TILE_CUH__
