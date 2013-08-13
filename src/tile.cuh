/***
  *  Project:
  *
  *  File: tile.cuh
  *  Created: Feb 02, 2013
  *  Modified: Tue 13 Aug 2013 11:59:57 AM PDT
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
			unsigned int final_size_;
			unsigned int tile_size_;

			// device buffers
			real_t* pattern_;
			cucomplex_t* vandermonde_;
			cucomplex_t* a_mat_;
			unsigned int* mask_mat_;
			cucomplex_t* f_mat_[2];
			real_t* mod_f_mat_[2];
			real_t* real_buff_d_;
			// host buffers
			cucomplex_t* complex_buff_h_;
			real_t* real_buff_h_;

			cufftHandle plan_;

			// temporary stuff
			cucomplex_t* virtual_a_mat_;

		public:
			__host__ GTile();
			__host__ ~GTile();

			__host__ bool init(real_t*, cucomplex_t*, real_t*, const unsigned int*,
								unsigned int, unsigned int, unsigned int, unsigned int);
			__host__ bool init_scale(real_t*, cucomplex_t*, real_t*, const unsigned int* mask,
									unsigned int, unsigned int, unsigned int);
			__host__ bool destroy_scale();

			__host__ bool compute_fft_mat(unsigned int);
			__host__ bool compute_virtual_fft_mat(unsigned int);
			__host__ cufftResult create_cufft_plan(cufftHandle&, cuFloatComplex*);
			__host__ cufftResult create_cufft_plan(cufftHandle&, cuDoubleComplex*);
			__host__ cufftResult execute_cufft(cufftHandle, cuFloatComplex*, cuFloatComplex*);
			__host__ cufftResult execute_cufft(cufftHandle, cuDoubleComplex*, cuDoubleComplex*);
			__host__ bool normalize_fft_mat(unsigned int, unsigned int);
			__host__ bool compute_mod_mat(unsigned int, unsigned int);
			__host__ bool copy_mod_mat(unsigned int);
			__host__ double compute_model_norm(unsigned int);
			__host__ double compute_chi2(unsigned int, real_t, real_t);
			__host__ bool compute_dft2(unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
										unsigned int, unsigned int);
			__host__ bool copy_f_mats_to_host(cucomplex_t*, real_t*, unsigned int, unsigned int);
			__host__ bool copy_model(mat_real_t&);
			__host__ bool copy_virtual_model(mat_real_t&);
	}; // class GTile

	// cuda kernels
	__global__ void compute_mod_mat_kernel(cucomplex_t*, unsigned int*, unsigned int, real_t*);
	__global__ void normalize_fft_mat_kernel(cucomplex_t*, unsigned int, unsigned int);
	__global__ void compute_model_norm_kernel(real_t*, unsigned int, unsigned int, real_t*);
	__global__ void compute_chi2_kernel(real_t*, real_t*, unsigned int, real_t, real_t*);
	__global__ void compute_dft2_kernel(cucomplex_t*, unsigned int, unsigned int, unsigned int,
										unsigned int, unsigned int, unsigned int, //cucomplex_t*,
										cucomplex_t*, cucomplex_t*);
	__global__ void compute_dft2_kernel_shared(cucomplex_t*, unsigned int, unsigned int, unsigned int,
										unsigned int, unsigned int, unsigned int, //cucomplex_t*,
										cucomplex_t*, cucomplex_t*);
	__global__ void compute_dft2_kernel_shared_opt(cucomplex_t*, unsigned int, unsigned int, unsigned int,
										unsigned int, unsigned int, unsigned int, //cucomplex_t*,
										cucomplex_t*, cucomplex_t*);
	__global__ void compute_dft2_kernel_shared_opt2(cucomplex_t*, unsigned int, unsigned int, unsigned int,
										unsigned int, unsigned int, unsigned int, //cucomplex_t*,
										cucomplex_t*, cucomplex_t*);
	__global__ void compute_dft2_kernel_shared_opt3(cucomplex_t*, unsigned int, unsigned int, unsigned int,
										unsigned int, unsigned int, unsigned int, //cucomplex_t*,
										cucomplex_t*, cucomplex_t*);
	__global__ void compute_dft2_kernel_shared_opt4(cucomplex_t*, unsigned int, unsigned int, unsigned int,
										unsigned int, unsigned int, unsigned int, //cucomplex_t*,
										cucomplex_t*, cucomplex_t*);
	__global__ void compute_dft2_kernel_shared_test(cucomplex_t*, unsigned int, unsigned int, unsigned int,
										unsigned int, unsigned int, unsigned int, //cucomplex_t*,
										cucomplex_t*, cucomplex_t*);
//	__global__ void update_fft_mat_kernel(cucomplex_t*, cucomplex_t*, unsigned int, cucomplex_t*);

} // namespace hir

#endif // __TILE_CUH__
