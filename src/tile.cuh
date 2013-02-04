/***
  *  Project:
  *
  *  File: tile.cuh
  *  Created: Feb 02, 2013
  *  Modified: Sun 03 Feb 2013 01:09:05 PM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef __TILE_CUH__
#define __TILE_CUH__

#include "typedefs.hpp"

namespace hir {

	class GTile {
		friend class Tile;

		private:
			dim3 block_dims_;
			dim3 grid_dims_;

		public:
			__host__ __device__ GTile() { }
			__host__ __device__ ~GTile() { }

	}; // class GTile

	__global__ void compute_mod_mat_kernel(cucomplex_t* inmat, unsigned int size, real_t* outmat);
	bool compute_mod_mat_cuda(cucomplex_t* f_mat, unsigned int size, real_t* mod_f_mat,
								unsigned int block_x, unsigned int block_y,
								unsigned int grid_x, unsigned int grid_y);
	__global__ void normalize_fft_mat_kernel(cucomplex_t* f_mat, unsigned int size,
												unsigned int num_particles);
	bool normalize_fft_mat_cuda(cucomplex_t* f_mat, unsigned int num_particles, unsigned int size,
								unsigned int block_x, unsigned int block_y,
								unsigned int grid_x, unsigned int grid_y);

} // namespace hir

#endif // __TILE_CUH__
