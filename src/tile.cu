/***
  *  Project:
  *
  *  File: tile.cu
  *  Created: Feb 02, 2013
  *  Modified: Fri 08 Mar 2013 07:51:30 PM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#include <iostream>
#include <thrust/reduce.h>

#include <woo/reduce/reduce.cuh>

#include "tile.cuh"
#include "typedefs.hpp"
#include "constants.hpp"
#include "utilities.cuh"

namespace hir {

	__host__ GTile::GTile():
		size_(0),
		pattern_(NULL), vandermonde_(NULL), a_mat_(NULL), mask_mat_(NULL),
		real_buff_d_(NULL),
		complex_buff_h_(NULL), real_buff_h_(NULL) {

		f_mat_[0] = NULL; f_mat_[1] = NULL;
		mod_f_mat_[0] = NULL; mod_f_mat_[1] = NULL;
	} // GTile::GTile()


	__host__ GTile::~GTile() {
		if(real_buff_h_ != NULL) delete[] real_buff_h_;
		if(complex_buff_h_ != NULL) delete[] complex_buff_h_;
		if(real_buff_d_ != NULL) cudaFree(real_buff_d_);
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
		cudaMalloc((void**) &real_buff_d_, size2 * sizeof(real_t));
		complex_buff_h_ = new (std::nothrow) cucomplex_t[size2];
		real_buff_h_ = new (std::nothrow) real_t[size2];
		if(pattern_ == NULL || vandermonde_ == NULL || a_mat_ == NULL ||
				f_mat_[0] == NULL || f_mat_[1] == NULL ||
				mod_f_mat_[0] == NULL || mod_f_mat_[1] == NULL ||
				mask_mat_ == NULL || real_buff_d_ == NULL) {
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
		cudaMemcpy(a_mat_, complex_buff_h_, size2 * sizeof(cucomplex_t), cudaMemcpyHostToDevice);
		cudaMemcpy(pattern_, pattern, size2 * sizeof(real_t), cudaMemcpyHostToDevice);
		cudaMemcpy(vandermonde_, vander, size2 * sizeof(cucomplex_t), cudaMemcpyHostToDevice);
		cudaMemcpy(mask_mat_, mask, size2 * sizeof(unsigned int), cudaMemcpyHostToDevice);
		return true;
	} // GTile::init()


	__host__ bool GTile::copy_f_mats_to_host(cucomplex_t* f_buff, real_t* mod_f_buff,
												unsigned int f_i, unsigned int mod_f_i) {
		unsigned int size2 = size_ * size_;
		cudaMemcpy(f_buff, f_mat_[f_i], size2 * sizeof(cucomplex_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(mod_f_buff, mod_f_mat_[mod_f_i], size2 * sizeof(real_t), cudaMemcpyDeviceToHost);
		return true;
	} // GTile::copy_f_mats_to_host()


	__host__ bool GTile::compute_fft_mat(unsigned int buff_i) {
        // create fft plan
        cufftHandle plan;
        cufftResult res;
        res = create_cufft_plan(plan, a_mat_);
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


	__host__ cufftResult GTile::create_cufft_plan(cufftHandle& plan, cuFloatComplex* a) {
        return cufftPlan2d(&plan, size_, size_, CUFFT_C2C);
	} // GTile::create_cufft_plan()


	__host__ cufftResult GTile::create_cufft_plan(cufftHandle& plan, cuDoubleComplex* a) {
        return cufftPlan2d(&plan, size_, size_, CUFFT_Z2Z);
	} // GTile::create_cufft_plan()


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
		compute_mod_mat_kernel <<< grid_dims_, block_dims_ >>> (f_mat_[src_i], mask_mat_, size_,
																mod_f_mat_[dst_i]);
		cudaThreadSynchronize();
		return true;
	} // GTile::compute_mod_mat()


	__host__ bool GTile::copy_mod_mat(unsigned int src_i) {
		cudaMemcpy(mod_f_mat_[1 - src_i], mod_f_mat_[src_i], size_ * size_ * sizeof(real_t),
					cudaMemcpyDeviceToDevice);
		cudaThreadSynchronize();	// not needed
		return true;
	} // GTile::copy_mod_mat()


	// reduction functor
	typedef struct {
		__host__ __device__
		real_t operator()(real_t a, real_t b) {
			return a + b;
		} // operator()()
	} plus_t;


	__host__ double GTile::compute_model_norm(unsigned int buff_i) {
		double model_norm = 0.0;
		unsigned int maxi = size_; // >> 1;
		compute_model_norm_kernel <<< grid_dims_, block_dims_ >>> (mod_f_mat_[buff_i], size_,
																	maxi, real_buff_d_);
		cudaThreadSynchronize();
		/*thrust::device_ptr<real_t> buff_p(real_buff_d_);
		thrust::plus<real_t> plus;
		model_norm = thrust::reduce(buff_p, buff_p + (maxi * maxi), 0.0, plus);
		*/
		plus_t plus_op;
		//model_norm = woo::cuda::reduce_multiple<real_t*, real_t, plus_t>(real_buff_d_, real_buff_d_ + (maxi * maxi),
		//													0.0, plus_op);
		model_norm = woo::cuda::reduce_single<real_t*, real_t, plus_t>(real_buff_d_, real_buff_d_ + (maxi * maxi),
															0.0, plus_op);
		return model_norm;
	} // GTile::compute_model_norm()


	__host__ double GTile::compute_chi2(unsigned int buff_i, real_t c_factor) {
		double chi2 = 0.0;
		compute_chi2_kernel <<< grid_dims_, block_dims_ >>> (pattern_, mod_f_mat_[buff_i], size_, c_factor,
															real_buff_d_);
		cudaThreadSynchronize();
		/*thrust::device_ptr<real_t> buff_p(real_buff_d_);
		thrust::plus<real_t> plus;
		chi2 = thrust::reduce(buff_p, buff_p + (size_ * size_), 0.0, plus);
		*/
		plus_t plus_op;
		//chi2 = woo::cuda::reduce_multiple<real_t*, real_t, plus_t>(real_buff_d_, real_buff_d_ + (size_ * size_),
		//													0.0, plus_op);
		chi2 = woo::cuda::reduce_single<real_t*, real_t, plus_t>(real_buff_d_, real_buff_d_ + (size_ * size_),
															0.0, plus_op);
		
		return chi2;
	} // GTile::compute_chi2()


	__host__ bool GTile::compute_dft2(unsigned int old_row, unsigned int old_col,
										unsigned int new_row, unsigned int new_col,
										unsigned int num_particles,
										unsigned int in_buff_i, unsigned int out_buff_i) {
		//compute_dft2_kernel <<< grid_dims_, block_dims_ >>> (vandermonde_, size_, old_row, old_col, new_row, new_col, num_particles, f_mat_[in_buff_i], f_mat_[out_buff_i]);
		compute_dft2_kernel_shared <<< grid_dims_, block_dims_ >>> (vandermonde_, size_, old_row, old_col, new_row, new_col, num_particles, f_mat_[in_buff_i], f_mat_[out_buff_i]);
		cudaThreadSynchronize();
		return true;
	} // GTile::compute_dft2()


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
	} // normalize_fft_mat_kernel()


	__global__ void compute_mod_mat_kernel(cucomplex_t* inmat, unsigned int* mask, unsigned int size,
											real_t* outmat) {
		unsigned int i_x = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int i_y = blockDim.y * blockIdx.y + threadIdx.y;
		if(i_x < size && i_y < size) {
			unsigned int index = size * i_y + i_x;
			cucomplex_t temp = inmat[index];
			//outmat[index] = temp.x * temp.x + temp.y * temp.y;
			outmat[index] = mask[index] * (temp.x * temp.x + temp.y * temp.y);
		} // if
	} // compute_mod_mat_kernel()


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
										unsigned int num_particles,
										cucomplex_t* fin, cucomplex_t* fout) {
		unsigned int i_x = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int i_y = blockDim.y * blockIdx.y + threadIdx.y;
		if(i_x < size && i_y < size) {
			unsigned int index = size * i_y + i_x;
			cucomplex_t new_temp = complex_mul(vandermonde[size * i_y + new_col],
												vandermonde[size * new_row + i_x]);
			cucomplex_t old_temp = complex_mul(vandermonde[size * i_y + old_col],
												vandermonde[size * old_row + i_x]);
			cucomplex_t dft_temp = complex_div((complex_sub(new_temp, old_temp)), (real_t)num_particles);
			fout[index] = complex_add(dft_temp, fin[index]);
		} // if
	} // compute_dft2_kernel()


	__global__ void compute_dft2_kernel_shared(cucomplex_t* vandermonde, unsigned int size,
							unsigned int old_row, unsigned int old_col,
							unsigned int new_row, unsigned int new_col,
							unsigned int num_particles,
							cucomplex_t* fin, cucomplex_t* fout) {
		// TODO: try subtiling also
		// TODO: try dynamic shared memory
		// TODO: try shared mem for output to coalesce writes
		unsigned int i_x = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int i_y = blockDim.y * blockIdx.y + threadIdx.y;
		unsigned int old_x = size * old_row + i_x;
		unsigned int old_y = size * i_y + old_col;
		unsigned int new_x = size * new_row + i_x;
		unsigned int new_y = size * i_y + new_col;

		// this basically makes copies of input vectors so that each thread
		// access different location in shared mem
		/*const unsigned int mat_size = CUDA_BLOCK_SIZE_X_ * CUDA_BLOCK_SIZE_Y_;
		__shared__ cucomplex_t vander_old_row[mat_size];
		__shared__ cucomplex_t vander_new_row[mat_size];
		__shared__ cucomplex_t vander_old_col[mat_size];
		__shared__ cucomplex_t vander_new_col[mat_size];
		unsigned int in_index = blockDim.x * threadIdx.y + threadIdx.x;
		if(i_x < size) {
			vander_old_row[in_index] = vandermonde[old_x];
			vander_new_row[in_index] = vandermonde[new_x];
		} // if
		if(i_y < size) {
			vander_old_col[in_index] = vandermonde[old_y];
			vander_new_col[in_index] = vandermonde[new_y];
		} // if*/

		__shared__ cucomplex_t vander_old_row[CUDA_BLOCK_SIZE_X_];
		__shared__ cucomplex_t vander_new_row[CUDA_BLOCK_SIZE_X_];
		__shared__ cucomplex_t vander_old_col[CUDA_BLOCK_SIZE_Y_];
		__shared__ cucomplex_t vander_new_col[CUDA_BLOCK_SIZE_Y_];
		// first row of threads load both rows
		if(threadIdx.y == 0 && i_x < size) {
			vander_old_row[threadIdx.x] = vandermonde[old_x];
			vander_new_row[threadIdx.x] = vandermonde[new_x];
		} // if
		// first col of threads load both cols
		if(threadIdx.x == 0 && i_y < size) {
			vander_old_col[threadIdx.y] = vandermonde[old_y];
			vander_new_col[threadIdx.y] = vandermonde[new_y];
		} // if

		__syncthreads();	// make sure all data is available

		unsigned int index = size * i_y + i_x;
		if(i_x < size && i_y < size) {
			/*cucomplex_t new_temp = complex_mul(vander_new_col[in_index],
								vander_new_row[in_index]);
			cucomplex_t old_temp = complex_mul(vander_old_col[in_index],
								vander_old_row[in_index]);*/
			cucomplex_t new_temp = complex_mul(vander_new_col[threadIdx.y],
								vander_new_row[threadIdx.x]);
			cucomplex_t old_temp = complex_mul(vander_old_col[threadIdx.y],
								vander_old_row[threadIdx.x]);
			cucomplex_t dft_temp = complex_div((complex_sub(new_temp, old_temp)),
								(real_t)num_particles);
			fout[index] = complex_add(dft_temp, fin[index]);
		} // if
	} // compute_dft2_kernel_shared()


} // namespace hir
