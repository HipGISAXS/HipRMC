/***
  *  Project:
  *
  *  File: tile.cu
  *  Created: Feb 02, 2013
  *  Modified: Mon 04 Feb 2013 03:32:03 PM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#include <iostream>
#include <thrust/reduce.h>

#include "tile.cuh"
#include "typedefs.hpp"
#include "constants.hpp"
#include "utilities.cuh"

namespace hir {

	__host__ GTile::GTile():
		size_(0),
		pattern_(NULL), vandermonde_(NULL), a_mat_(NULL), mask_mat_(NULL), dft_mat_(NULL),
		real_buff_d_(NULL),
		complex_buff_h_(NULL), real_buff_h_(NULL) {

		f_mat_[0] = NULL; f_mat_[1] = NULL;
		mod_f_mat_[0] = NULL; mod_f_mat_[1] = NULL;
	} // GTile::GTile()


	__host__ GTile::~GTile() {
		if(real_buff_h_ != NULL) delete[] real_buff_h_;
		if(complex_buff_h_ != NULL) delete[] complex_buff_h_;
		if(real_buff_d_ != NULL) cudaFree(real_buff_d_);
		if(dft_mat_ != NULL) cudaFree(dft_mat_);
		if(mask_mat_ != NULL) cudaFree(mask_mat_);
		if(mod_f_mat_[1] != NULL) cudaFree(mod_f_mat_[1]);
		if(mod_f_mat_[0] != NULL) cudaFree(mod_f_mat_[0]);
		if(f_mat_[1] != NULL) cudaFree(f_mat_[1]);
		if(f_mat_[0] != NULL) cudaFree(f_mat_[0]);
		if(a_mat_ != NULL) cudaFree(a_mat_);
		if(vandermonde_ != NULL) cudaFree(vandermonde_);
		if(pattern_ != NULL) cudaFree(pattern_);
	} // GTile::GTile()


	__host__ bool GTile::init(real_t* pattern, cucomplex_t* vander, real_t* a,
			const unsigned int* mask, unsigned int size,
			unsigned int block_x = CUDA_BLOCK_SIZE_X_, unsigned int block_y = CUDA_BLOCK_SIZE_Y_) {
		size_ = size;
		unsigned int size2 = size_ * size_;
		unsigned int grid_x = (unsigned int) ceil((real_t) size_ / block_x);
		unsigned int grid_y = (unsigned int) ceil((real_t) size_ / block_y);
		block_dims_ = dim3(block_x, block_y, 1);
		grid_dims_ = dim3(grid_x, grid_y, 1);

		// allocate device memory and host buffer memory
		cudaMalloc((void**) &pattern_, size2 * sizeof(real_t));
		cudaMalloc((void**) &vandermonde_, size2 * sizeof(cucomplex_t));
		cudaMalloc((void**) &a_mat_, size2 * sizeof(cucomplex_t));
		cudaMalloc((void**) &f_mat_[0], size2 * sizeof(cucomplex_t));
		cudaMalloc((void**) &f_mat_[1], size2 * sizeof(cucomplex_t));
		cudaMalloc((void**) &mod_f_mat_[0], size2 * sizeof(real_t));
		cudaMalloc((void**) &mod_f_mat_[1], size2 * sizeof(real_t));
		cudaMalloc((void**) &mask_mat_, size2 * sizeof(unsigned int));
		cudaMalloc((void**) &dft_mat_, size2 * sizeof(cucomplex_t));
		cudaMalloc((void**) &real_buff_d_, size2 * sizeof(real_t));
		complex_buff_h_ = new (std::nothrow) cucomplex_t[size2];
		real_buff_h_ = new (std::nothrow) real_t[size2];
		if(pattern_ == NULL || vandermonde_ == NULL || a_mat_ == NULL ||
				f_mat_[0] == NULL || f_mat_[1] == NULL ||
				mod_f_mat_[0] == NULL || mod_f_mat_[1] == NULL ||
				mask_mat_ == NULL || dft_mat_ == NULL || real_buff_d_ == NULL) {
			std::cerr << "error: failed to allocate device memory." << std::endl;
			return false;
		} // if
		if(complex_buff_h_ == NULL || real_buff_h_ == NULL) {
			std::cerr << "error: failed to allocate host buffer memory." << std::endl;
			return false;
		} // if
		// set a_mat_
		for(int i = 0; i < size2; ++ i) {
			complex_buff_h_[i].x = a[i];
			complex_buff_h_[i].y = 0.0;
		} // for
		// copy data to device
		cudaMemcpy(pattern_, pattern, size2 * sizeof(real_t), cudaMemcpyHostToDevice);
		cudaMemcpy(vandermonde_, vander, size2 * sizeof(cucomplex_t), cudaMemcpyHostToDevice);
		cudaMemcpy(a_mat_, complex_buff_h_, size2 * sizeof(cucomplex_t), cudaMemcpyHostToDevice);
		cudaMemcpy(mask_mat_, mask, size2 * sizeof(unsigned int), cudaMemcpyHostToDevice);
		return true;
	} // GTile::init()


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
	} // GTile::compute_mod_mat()


	__host__ bool GTile::mask_mat(unsigned int buff_i) {
		mask_mat_kernel <<< grid_dims_, block_dims_ >>> (mask_mat_, size_, mod_f_mat_[buff_i]);
		cudaThreadSynchronize();
		return true;
	} // GTile::mask_mat()


	__host__ bool GTile::copy_mod_mat(unsigned int src_i) {
		cudaMemcpy(mod_f_mat_[1 - src_i], mod_f_mat_[src_i], size_ * size_ * sizeof(real_t),
					cudaMemcpyDeviceToDevice);
		cudaThreadSynchronize();	// not needed
		return true;
	} // GTile::copy_mod_mat()


	__host__ double GTile::compute_model_norm(unsigned int buff_i) {
		// TODO: perform a nicer reduction operation ...
		double model_norm = 0.0;
		unsigned int maxi = size_ >> 1;
		compute_model_norm_kernel <<< grid_dims_, block_dims_ >>> (mod_f_mat_[buff_i], size_,
																	maxi, real_buff_d_);
		cudaThreadSynchronize();
		thrust::device_ptr<real_t> buff_p(real_buff_d_);
		thrust::plus<real_t> plus;
		model_norm = thrust::reduce(buff_p, buff_p + (maxi * maxi), 0.0, plus);
		return model_norm;
	} // GTile::compute_model_norm()


	__host__ double GTile::compute_chi2(unsigned int buff_i, real_t c_factor) {
		double chi2 = 0.0;
		compute_chi2_kernel <<< grid_dims_, block_dims_ >>> (pattern_, mod_f_mat_[buff_i], size_, c_factor,
															real_buff_d_);
		cudaThreadSynchronize();
		thrust::device_ptr<real_t> buff_p(real_buff_d_);
		thrust::plus<real_t> plus;
		chi2 = thrust::reduce(buff_p, buff_p + (size_ * size_), 0.0, plus);
		return chi2;
	} // GTile::compute_chi2()


	__host__ bool GTile::compute_dft2(unsigned int old_row, unsigned int old_col,
										unsigned int new_row, unsigned int new_col,
										unsigned int num_particles) {
		compute_dft2_kernel <<< grid_dims_, block_dims_ >>> (vandermonde_, size_, old_row, old_col,
															new_row, new_col, num_particles, vandermonde_);
		cudaThreadSynchronize();
		return true;
	} // GTile::compute_dft2()


	__host__ bool GTile::update_fft_mat(unsigned int in_buff_i, unsigned int out_buff_i) {
		update_fft_mat_kernel <<< grid_dims_, block_dims_ >>> (dft_mat_, f_mat_[in_buff_i], size_,
																f_mat_[out_buff_i]);
		cudaThreadSynchronize();
		return true;
	} // GTile::update_fft_mat()


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


	__global__ void mask_mat_kernel(unsigned int* mask, unsigned int size, real_t* mat) {
		unsigned int i_x = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int i_y = blockDim.y * blockIdx.y + threadIdx.y;
		if(i_x < size && i_y < size) {
			unsigned int index = size * i_y + i_x;
			mat[index] *= mask[index];
		} // if
	} // mask_mat_kernel()


	__global__ void compute_model_norm_kernel(real_t* inmat, unsigned int size,
												unsigned int size2, real_t* outmat) {
		unsigned int i_x = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int i_y = blockDim.y * blockIdx.y + threadIdx.y;
		if(i_x < size2 && i_y < size2) {
			unsigned int index_in = size * i_y + i_x;
			unsigned int index_out = size2 * i_y + i_x;
			outmat[index_out] = inmat[index_in] * (i_x + 1);
		} // if
	} // compute_model_norm_kernel()


	__global__ void compute_chi2_kernel(real_t* pattern, real_t* mod_mat, unsigned int size,
										real_t c_factor, real_t* out) {
		unsigned int i_x = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int i_y = blockDim.y * blockIdx.y + threadIdx.y;
		if(i_x < size && i_y < size) {
			unsigned int index = size * i_y + i_x;
			real_t temp = pattern[index] - mod_mat[index] * c_factor;
			out[index] = temp * temp;
		} // if
	} // compute_chi2_kernel()


	__global__ void compute_dft2_kernel(cucomplex_t* vandermonde, unsigned int size,
										unsigned int old_row, unsigned int old_col,
										unsigned int new_row, unsigned int new_col,
										unsigned int num_particles, cucomplex_t* dft_mat) {
		unsigned int i_x = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int i_y = blockDim.y * blockIdx.y + threadIdx.y;
		if(i_x < size && i_y < size) {
			unsigned int index = size * i_y + i_x;
			cucomplex_t new_temp = complex_mul(vandermonde[size * i_y + new_col],
												vandermonde[size * new_row + i_x]);
			cucomplex_t old_temp = complex_mul(vandermonde[size * i_y + old_col],
												vandermonde[size * old_row + i_x]);
			dft_mat[index] = complex_div((complex_sub(new_temp, old_temp)), (real_t)num_particles);
		} // if
	} // compute_dft2_kernel()


	__global__ void update_fft_mat_kernel(cucomplex_t* dft, cucomplex_t* fin, unsigned int size,
											cucomplex_t* fout) {
		unsigned int i_x = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int i_y = blockDim.y * blockIdx.y + threadIdx.y;
		if(i_x < size && i_y < size) {
			unsigned int index = size * i_y + i_x;
			fout[index] = complex_add(dft[index], fin[index]);
		} // if
	} // update_fft_mat_kernel()

} // namespace hir
