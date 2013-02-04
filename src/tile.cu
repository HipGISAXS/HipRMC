/***
  *  Project:
  *
  *  File: tile.cu
  *  Created: Feb 02, 2013
  *  Modified: Sun 03 Feb 2013 01:09:23 PM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#include <iostream>
#include <cuComplex.h>

#include "tile.cuh"
#include "typedefs.hpp"
#include "constants.hpp"
#include "utilities.cuh"

namespace hir {


	__global__ void compute_mod_mat_kernel(cucomplex_t* inmat, unsigned int size, real_t* outmat) {
		unsigned int i_x = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int i_y = blockDim.y * blockIdx.y + threadIdx.y;

		if(i_x < size && i_y < size) {
			unsigned int index = size * i_y + i_x;
			cucomplex_t temp = inmat[index];
			outmat[index] = temp.x * temp.x + temp.y * temp.y;
		} // if
	} // compute_mod_mat_kernel()

	bool compute_mod_mat_cuda(cucomplex_t* f_mat, unsigned int size, real_t* mod_f_mat,
								unsigned int block_x, unsigned int block_y,
								unsigned int grid_x, unsigned int grid_y) {
		std::cout << "*" << std::endl;
		// using 2D for now ... see if 1D performs better ...
		compute_mod_mat_kernel <<< dim3(grid_x, grid_y, 1), dim3(block_x, block_y, 1) >>>
								(f_mat, size, mod_f_mat);
		cudaThreadSynchronize();
		return true;
	} // compute_mod_mat_cuda()


	__global__ void normalize_fft_mat_kernel(cucomplex_t* f_mat, unsigned int size,
												unsigned int num_particles) {
		unsigned int i_x = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int i_y = blockDim.y * blockIdx.y + threadIdx.y;
		if(i_x < size && i_y < size) {
			f_mat[size * i_y + i_x] = make_cuComplex(f_mat[size * i_y + i_x].x / num_particles,
													f_mat[size * i_y + i_x].y / num_particles);
		} // if
	} // compute_mod_mat_kernel()

	bool normalize_fft_mat_cuda(cucomplex_t* f_mat, unsigned int num_particles, unsigned int size,
								unsigned int block_x, unsigned int block_y,
								unsigned int grid_x, unsigned int grid_y) {
		//normalize_fft_mat_kernel<real_t, cucomplex_t>
		normalize_fft_mat_kernel
								<<< dim3(grid_x, grid_y, 1), dim3(block_x, block_y, 1) >>>
								(f_mat, size, num_particles);
		cudaThreadSynchronize();
		return true;
	} // normalize_fft_mat_cuda()

} // namespace hir
