/***
  *  Project:
  *
  *  File: tile.cu
  *  Created: Feb 02, 2013
  *  Modified: Mon 04 Feb 2013 11:41:08 AM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#include <iostream>

#include "tile.cuh"
#include "typedefs.hpp"
#include "constants.hpp"
#include "utilities.cuh"

namespace hir {

	__host__ GTile::GTile():
		size_(0),
		a_mat_(NULL),
		complex_buff_h_(NULL), real_buff_h_(NULL) {
		f_mat_[0] = NULL; f_mat_[1] = NULL;
		mod_f_mat_[0] = NULL; mod_f_mat_[1] = NULL;
	} // GTile::GTile()


	__host__ GTile::~GTile() {
		if(real_buff_h_ != NULL) delete[] real_buff_h_;
		if(complex_buff_h_ != NULL) delete[] complex_buff_h_;
		if(mod_f_mat_[1] != NULL) cudaFree(mod_f_mat_[1]);
		if(mod_f_mat_[0] != NULL) cudaFree(mod_f_mat_[0]);
		if(f_mat_[1] != NULL) cudaFree(f_mat_[1]);
		if(f_mat_[0] != NULL) cudaFree(f_mat_[0]);
		if(a_mat_ != NULL) cudaFree(a_mat_);
	} // GTile::GTile()


	__host__ bool GTile::init(unsigned int size,
			unsigned int block_x = CUDA_BLOCK_SIZE_X_, unsigned int block_y = CUDA_BLOCK_SIZE_Y_) {
		size_ = size;
		unsigned int size2 = size_ * size_;
		unsigned int grid_x = (unsigned int) ceil((real_t) size_ / block_x);
		unsigned int grid_y = (unsigned int) ceil((real_t) size_ / block_y);
		block_dims_ = dim3(block_x, block_y, 1);
		grid_dims_ = dim3(grid_x, grid_y, 1);

		// allocate device memory and host buffer memory
		cudaMalloc((void**) &a_mat_, size2 * sizeof(cucomplex_t));
		cudaMalloc((void**) &f_mat_[0], size2 * sizeof(cucomplex_t));
		cudaMalloc((void**) &f_mat_[1], size2 * sizeof(cucomplex_t));
		cudaMalloc((void**) &mod_f_mat_[0], size2 * sizeof(real_t));
		cudaMalloc((void**) &mod_f_mat_[1], size2 * sizeof(real_t));
		complex_buff_h_ = new (std::nothrow) cucomplex_t[size2];
		real_buff_h_ = new (std::nothrow) real_t[size2];
		if(a_mat_ == NULL || f_mat_[0] == NULL || f_mat_[1] == NULL ||
				mod_f_mat_[0] == NULL || mod_f_mat_[1] == NULL) {
			std::cerr << "error: failed to allocate device memory." << std::endl;
			return false;
		} // if
		if(complex_buff_h_ == NULL || real_buff_h_ == NULL) {
			std::cerr << "error: failed to allocate host buffer memory." << std::endl;
			return false;
		} // if
		return true;
	} // GTile::init()


	__host__ bool GTile::set_a_mat(real_t* a) {
		unsigned int size2 = size_ * size_;
		for(int i = 0; i < size2; ++ i) {
			complex_buff_h_[i].x = a[i];
			complex_buff_h_[i].y = 0.0;
		} // for
		cudaMemcpy(a_mat_, complex_buff_h_, size2 * sizeof(cucomplex_t), cudaMemcpyHostToDevice);
		return true;
	} // GTile::set_a_mat()


	__host__ bool GTile::compute_fft_mat(unsigned int buff_i) {
        // create fft plan
        cufftHandle plan;
        cufftResult res;
        res = cufftPlan2d(&plan, size_, size_, CUFFT_C2C);
        if(res != CUFFT_SUCCESS) {
            std::cerr << "error: " << res << ": fft plan could not be created" << std::endl;
            return false;
        } // if
		res = execute_cufft(plan, a_mat_, f_mat_[buff_i]);
        if(res != CUFFT_SUCCESS) {
            std::cerr << "error: " << res << ": fft could not be executed" << std::endl;
            return false;
        } // if
        cudaThreadSynchronize();
        // destroy fft plan
        cufftDestroy(plan);
        return true;
    } // GTile::compute_fft_mat()


	__host__ cufftResult GTile::execute_cufft(cufftHandle plan, cuFloatComplex* a, cuFloatComplex* f) {
        return cufftExecC2C(plan, a, f, CUFFT_FORWARD);
	} // GTile::execute_cufft()


	__host__ cufftResult GTile::execute_cufft(cufftHandle plan, cuDoubleComplex* a, cuDoubleComplex* f) {
        return cufftExecZ2Z(plan, a, f, CUFFT_FORWARD);
	} // GTile::execute_cufft()


	__host__ bool GTile::normalize_fft_mat(unsigned int buff_i, unsigned int num_particles) {
		normalize_fft_mat_kernel <<< grid_dims_, block_dims_ >>> (f_mat_[buff_i], size_, num_particles);
		cudaThreadSynchronize();
		return true;
	} // GTile::normalize_fft_mat_cuda()


	__host__ bool GTile::compute_mod_mat(unsigned int src_i, unsigned int dst_i) {
		compute_mod_mat_kernel <<< grid_dims_, block_dims_ >>> (f_mat_[src_i], size_, mod_f_mat_[dst_i]);
		cudaThreadSynchronize();
		return true;
	} // compute_mod_mat_cuda()


	// ////
	// CUDA kernels
	// ////


	__global__ void normalize_fft_mat_kernel(cucomplex_t* f_mat, unsigned int size,
												unsigned int num_particles) {
		unsigned int i_x = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int i_y = blockDim.y * blockIdx.y + threadIdx.y;
		if(i_x < size && i_y < size) {
			f_mat[size * i_y + i_x] = make_cuComplex(f_mat[size * i_y + i_x].x / num_particles,
													f_mat[size * i_y + i_x].y / num_particles);
		} // if
	} // compute_mod_mat_kernel()


	__global__ void compute_mod_mat_kernel(cucomplex_t* inmat, unsigned int size, real_t* outmat) {
		unsigned int i_x = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int i_y = blockDim.y * blockIdx.y + threadIdx.y;
		if(i_x < size && i_y < size) {
			unsigned int index = size * i_y + i_x;
			cucomplex_t temp = inmat[index];
			outmat[index] = temp.x * temp.x + temp.y * temp.y;
		} // if
	} // compute_mod_mat_kernel()


} // namespace hir
