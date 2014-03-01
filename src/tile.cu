/***
  *  Project:
  *
  *  File: tile.cu
  *  Created: Feb 02, 2013
  *  Modified: Sat 01 Mar 2014 07:26:20 AM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#include <iostream>
//#include <thrust/reduce.h>
//#include <thrust/device_ptr.h>

#include "woo/reduce/reduce.cuh"

#include <nvToolsExt.h>
#include <cuda_profiler_api.h>

#include "tile.cuh"
#include "typedefs.hpp"
#include "constants.hpp"
#include "utilities.cuh"

namespace hir {

	__host__ GTile::GTile():
		final_size_(0), tile_size_(0),
		pattern_(NULL), vandermonde_(NULL), a_mat_(NULL), mask_mat_(NULL),
		#ifndef USE_DFT
		virtual_a_mat_(NULL),
		#endif
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
		#ifndef USE_DFT
		if(virtual_a_mat_ != NULL) cudaFree(virtual_a_mat_);
		#endif
		if(a_mat_ != NULL) cudaFree(a_mat_);
		if(vandermonde_ != NULL) cudaFree(vandermonde_);
		if(pattern_ != NULL) cudaFree(pattern_);
	} // GTile::GTile()


	//__host__ bool GTile::init(real_t* pattern, cucomplex_t* vander, real_t* a,
	//		const unsigned int* mask, unsigned int size, unsigned int tile_size,
	//		unsigned int block_x = CUDA_BLOCK_SIZE_X_, unsigned int block_y = CUDA_BLOCK_SIZE_Y_) {
	__host__ bool GTile::init(real_t* a, unsigned int size, unsigned int tile_size,
			unsigned int block_x = CUDA_BLOCK_SIZE_X_, unsigned int block_y = CUDA_BLOCK_SIZE_Y_) {
		// NOT USED stuff: pattern, vander, a, mask ...
		final_size_ = size;
		tile_size_ = tile_size;
		unsigned int size2 = final_size_ * final_size_;

		// allocate device memory and host buffer memory for full size once
		cudaMalloc((void**) &pattern_, size2 * sizeof(real_t));
		cudaMalloc((void**) &vandermonde_, size2 * sizeof(cucomplex_t));
		cudaMalloc((void**) &a_mat_, size2 * sizeof(cucomplex_t));
		#ifndef USE_DFT
		cudaMalloc((void**) &virtual_a_mat_, size2 * sizeof(cucomplex_t));
		#endif
		cudaMalloc((void**) &f_mat_[0], size2 * sizeof(cucomplex_t));
		cudaMalloc((void**) &f_mat_[1], size2 * sizeof(cucomplex_t));
		cudaMalloc((void**) &mod_f_mat_[0], size2 * sizeof(real_t));
		cudaMalloc((void**) &mod_f_mat_[1], size2 * sizeof(real_t));
		cudaMalloc((void**) &mask_mat_, size2 * sizeof(unsigned int));
		cudaMalloc((void**) &real_buff_d_, size2 * sizeof(real_t));
		complex_buff_h_ = new (std::nothrow) cucomplex_t[size2];
		real_buff_h_ = new (std::nothrow) real_t[size2];
		if(pattern_ == NULL || vandermonde_ == NULL || a_mat_ == NULL ||
				#ifndef USE_DFT
				virtual_a_mat_ == NULL ||
				#endif
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

		return true;
	} // GTile::init()


	__host__ bool GTile::init_scale(real_t* pattern, cucomplex_t* vander, real_t* a,
									const unsigned int* mask, unsigned int tile_size,
									unsigned int block_x = CUDA_BLOCK_SIZE_X_,
									unsigned int block_y = CUDA_BLOCK_SIZE_Y_) {
		tile_size_ = tile_size;
		unsigned int tsize2 = tile_size_ * tile_size_;

		// copy current data to device
		for(int i = 0; i < tsize2; ++ i) {
			complex_buff_h_[i].x = a[i];
			complex_buff_h_[i].y = 0.0;
		} // for
		cudaMemcpy(a_mat_, complex_buff_h_, tsize2 * sizeof(cucomplex_t), cudaMemcpyHostToDevice);
		cudaMemcpy(pattern_, pattern, tsize2 * sizeof(real_t), cudaMemcpyHostToDevice);
		cudaMemcpy(vandermonde_, vander, tsize2 * sizeof(cucomplex_t), cudaMemcpyHostToDevice);
		cudaMemcpy(mask_mat_, mask, tsize2 * sizeof(unsigned int), cudaMemcpyHostToDevice);

		unsigned int grid_x = (unsigned int) ceil((real_t) tile_size_ / block_x);
		unsigned int grid_y = (unsigned int) ceil((real_t) tile_size_ / block_y);
		block_dims_ = dim3(block_x, block_y, 1);
		grid_dims_ = dim3(grid_x, grid_y, 1);

        cufftResult res = create_cufft_plan(plan_, a_mat_);
        if(res != CUFFT_SUCCESS) {
            std::cerr << "error: " << res << ": fft plan could not be created" << std::endl;
            return false;
        } // if

		return true;
	} // GTile::init_scale()


	__host__ bool GTile::destroy_scale() {
        // destroy fft plan
        cufftDestroy(plan_);
		return true;
	} // GTile::destroy_scale()


	__host__ bool GTile::copy_mod_mat(unsigned int src_i) {
		cudaMemcpy(mod_f_mat_[1 - src_i], mod_f_mat_[src_i], tile_size_ * tile_size_ * sizeof(real_t),
					cudaMemcpyDeviceToDevice);
		cudaThreadSynchronize();	// not needed
		return true;
	} // GTile::copy_mod_mat()


	__host__ bool GTile::copy_model(mat_real_t& a) {
		for(int i = 0; i < tile_size_; ++ i) {
			for(int j = 0; j < tile_size_; ++ j) {
				complex_t temp = a(i, j);
				complex_buff_h_[tile_size_ * i + j].x = temp.real();
				complex_buff_h_[tile_size_ * i + j].y = 0.0;
			} // for j
		} // for i
		cudaMemcpy(a_mat_, complex_buff_h_, tile_size_ * tile_size_ * sizeof(cucomplex_t),
					cudaMemcpyHostToDevice);
		return true;
	} // GTile::copy_model()


	__host__ bool GTile::copy_f_mats_to_host(cucomplex_t* f_buff, real_t* mod_f_buff,
												unsigned int f_i, unsigned int mod_f_i) {
		unsigned int size2 = tile_size_ * tile_size_;
		cudaMemcpy(f_buff, f_mat_[f_i], size2 * sizeof(cucomplex_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(mod_f_buff, mod_f_mat_[mod_f_i], size2 * sizeof(real_t), cudaMemcpyDeviceToHost);
		//for(int i = 0; i < tile_size_; ++ i) {
		//	for(int j = 0; j < tile_size_; ++ j) {
		//		std::cout << f_buff[tile_size_ * i + j].x << "+" << f_buff[tile_size_ * i + j].y << " ";
		//	} // for
		//	std::cout << std::endl;
		//} // for 
		//std::cout << " +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ " << std::endl;
		//for(int i = 0; i < tile_size_; ++ i) {
		//	for(int j = 0; j < tile_size_; ++ j) {
		//		std::cout << mod_f_buff[tile_size_ * i + j] << " ";
		//	} // for
		//	std::cout << std::endl;
		//} // for 
		return true;
	} // GTile::copy_f_mats_to_host()


	#ifndef USE_DFT

	__host__ bool GTile::copy_virtual_model(mat_real_t& a) {
		for(int i = 0; i < tile_size_; ++ i) {
			for(int j = 0; j < tile_size_; ++ j) {
				complex_t temp = a(i, j);
				complex_buff_h_[tile_size_ * i + j].x = temp.real();
				complex_buff_h_[tile_size_ * i + j].y = 0.0;
			} // for j
		} // for i
		cudaMemcpy(virtual_a_mat_, complex_buff_h_, tile_size_ * tile_size_ * sizeof(cucomplex_t),
					cudaMemcpyHostToDevice);
		return true;
	} // GTile::copy_model()


	__host__ bool GTile::compute_virtual_fft_mat(unsigned int buff_i) {
        cufftResult res = execute_cufft(plan_, virtual_a_mat_, f_mat_[buff_i]);
        if(res != CUFFT_SUCCESS) {
            std::cerr << "error: " << res << ": fft could not be executed" << std::endl;
            return false;
        } // if
        cudaThreadSynchronize();
        return true;
    } // GTile::compute_virtual_fft_mat()

	#endif // USE_DFT


	__host__ bool GTile::compute_fft_mat(unsigned int buff_i) {
		cufftResult res = execute_cufft(plan_, a_mat_, f_mat_[buff_i]);
        if(res != CUFFT_SUCCESS) {
            std::cerr << "error: " << res << ": fft could not be executed" << std::endl;
            return false;
        } // if
        cudaThreadSynchronize();
        return true;
    } // GTile::compute_fft_mat()


	// specializations for float and double

	__host__ cufftResult GTile::create_cufft_plan(cufftHandle& plan, cuFloatComplex* a) {
        return cufftPlan2d(&plan, tile_size_, tile_size_, CUFFT_C2C);
	} // GTile::create_cufft_plan()


	__host__ cufftResult GTile::create_cufft_plan(cufftHandle& plan, cuDoubleComplex* a) {
        return cufftPlan2d(&plan, tile_size_, tile_size_, CUFFT_Z2Z);
	} // GTile::create_cufft_plan()


	__host__ cufftResult GTile::execute_cufft(cufftHandle plan, cuFloatComplex* a, cuFloatComplex* f) {
        return cufftExecC2C(plan, a, f, CUFFT_FORWARD);
	} // GTile::execute_cufft()


	__host__ cufftResult GTile::execute_cufft(cufftHandle plan, cuDoubleComplex* a, cuDoubleComplex* f) {
        return cufftExecZ2Z(plan, a, f, CUFFT_FORWARD);
	} // GTile::execute_cufft()


	__host__ bool GTile::normalize_fft_mat(unsigned int buff_i, unsigned int num_particles) {
		normalize_fft_mat_kernel <<< grid_dims_, block_dims_ >>> (f_mat_[buff_i], tile_size_, num_particles);
		cudaThreadSynchronize();
		return true;
	} // GTile::normalize_fft_mat_cuda()


	__host__ bool GTile::compute_mod_mat(unsigned int src_i, unsigned int dst_i) {
		compute_mod_mat_kernel <<< grid_dims_, block_dims_ >>> (f_mat_[src_i], mask_mat_, tile_size_,
																mod_f_mat_[dst_i]);
		cudaThreadSynchronize();
		return true;
	} // GTile::compute_mod_mat()


	typedef struct {
		__host__ __device__
		real_t operator()(real_t a, real_t b) {
			return ((a < b) ? a : b);
		} // operator()()
	} min_t;

	typedef struct {
		__host__ __device__
		real_t operator()(real_t a, real_t b) {
			return ((a > b) ? a : b);
		} // operator()()
	} max_t;


	__host__ bool GTile::normalize_mod_mat(unsigned int buff_i) {
		min_t min_op;
		max_t max_op;
		//std::cout << "****************************************************" << std::endl;
		//print_modf_mat(buff_i);
		//std::cout << "****************************************************" << std::endl;
		real_t min_val = woo::cuda::reduce_single<real_t*, real_t, min_t>(mod_f_mat_[buff_i],
										mod_f_mat_[buff_i] + (tile_size_ * tile_size_), 1e10, min_op);
		//std::cout << "****************************************************" << std::endl;
		//print_modf_mat(buff_i);
		//std::cout << "****************************************************" << std::endl;
		real_t max_val = woo::cuda::reduce_single<real_t*, real_t, max_t>(mod_f_mat_[buff_i],
										mod_f_mat_[buff_i] + (tile_size_ * tile_size_), 0.0, max_op);
		//std::cout << "MIN: " << min_val << ", MAX: " << max_val << std::endl;
		normalize_mod_mat_kernel <<< grid_dims_, block_dims_ >>> (mod_f_mat_[buff_i], tile_size_,
										min_val, max_val);
		cudaThreadSynchronize();
		return true;
	} // GTile::normalize_mod_mat()


	// reduction functor
	typedef struct {
		__host__ __device__
		real_t operator()(real_t a, real_t b) {
			return a + b;
		} // operator()()
	} plus_t;


	__host__ double GTile::compute_model_norm(unsigned int buff_i) {
		double model_norm = 0.0;
		unsigned int maxi = tile_size_;
		//compute_model_norm_kernel <<< grid_dims_, block_dims_ >>> (mod_f_mat_[buff_i], tile_size_,
		//															maxi, real_buff_d_);
		//cudaThreadSynchronize();
		/*thrust::device_ptr<real_t> buff_p(real_buff_d_);
		thrust::plus<real_t> plus;
		model_norm = thrust::reduce(buff_p, buff_p + (maxi * maxi), 0.0, plus);
		*/
		plus_t plus_op;
		//model_norm = woo::cuda::reduce_multiple<real_t*, real_t, plus_t>(real_buff_d_,
		//													real_buff_d_ + (maxi * maxi),
		//													0.0, plus_op);
		model_norm = woo::cuda::reduce_single<real_t*, real_t, plus_t>(mod_f_mat_[buff_i],
												mod_f_mat_[buff_i] + (maxi * maxi), 0.0, plus_op);
		return model_norm;
	} // GTile::compute_model_norm()


	__host__ double GTile::compute_chi2(unsigned int buff_i, real_t c_factor, real_t base_norm) {
		double chi2 = 0.0;
		compute_chi2_kernel <<< grid_dims_, block_dims_ >>> (pattern_, mod_f_mat_[buff_i], tile_size_,
															c_factor, real_buff_d_);
		cudaThreadSynchronize();
		//thrust::device_ptr<real_t> buff_p(real_buff_d_);
		//thrust::plus<real_t> plus;
		//chi2 = thrust::reduce(buff_p, buff_p + (tile_size_ * tile_size_), 0.0, plus);
		
		plus_t plus_op;
		//chi2 = woo::cuda::reduce_multiple<real_t*, real_t, plus_t>(real_buff_d_,
		//													real_buff_d_ + (tile_size_ * tile_size_),
		//													0.0, plus_op);
		chi2 = woo::cuda::reduce_single<real_t*, real_t, plus_t>(real_buff_d_,
										real_buff_d_ + (tile_size_ * tile_size_), 0.0, plus_op);
		return chi2;
	} // GTile::compute_chi2()


	#ifdef USE_DFT
	__host__ bool GTile::compute_dft2(unsigned int old_row, unsigned int old_col,
										unsigned int new_row, unsigned int new_col,
										unsigned int num_particles,
										unsigned int in_buff_i, unsigned int out_buff_i) {
		//nvtxRangeId_t nvtx0 = nvtxRangeStart("dft2_compute");
		cudaProfilerStart();
		compute_dft2_kernel <<< grid_dims_, block_dims_ >>>
								(vandermonde_, tile_size_, old_row, old_col, new_row, new_col,
								num_particles, f_mat_[in_buff_i], f_mat_[out_buff_i]);
		//compute_dft2_kernel_shared <<< grid_dims_, block_dims_ >>>
		//						(vandermonde_, tile_size_, old_row, old_col, new_row, new_col,
		//						num_particles, f_mat_[in_buff_i], f_mat_[out_buff_i]);
		//compute_dft2_kernel_shared_opt2 <<< grid_dims_, block_dims_ >>>
		//						(vandermonde_, tile_size_, old_row, old_col, new_row, new_col,
		//						num_particles, f_mat_[in_buff_i], f_mat_[out_buff_i]);
		//compute_dft2_kernel_shared_opt3 <<< grid_dims_, block_dims_ >>>
		//						(vandermonde_, tile_size_, old_row, old_col, new_row, new_col,
		//						num_particles, f_mat_[in_buff_i], f_mat_[out_buff_i]);
		unsigned int block_x = CUDA_BLOCK_SIZE_X_;
		unsigned int block_y = CUDA_BLOCK_SIZE_Y_;
		unsigned int grid_x = (unsigned int) ceil((real_t) tile_size_ / block_x);
		unsigned int grid_y = (unsigned int) ceil((real_t) tile_size_ / (block_y * CUDA_DFT2_SUBTILES_));
		dim3 block_dims = dim3(block_x, block_y, 1);
		dim3 grid_dims = dim3(grid_x, grid_y, 1);
		//compute_dft2_kernel_shared_opt4 <<< grid_dims_, block_dims_ >>>
		//						(vandermonde_, tile_size_, old_row, old_col, new_row, new_col,
		//						num_particles, f_mat_[in_buff_i], f_mat_[out_buff_i]);
		cudaProfilerStop();
		//nvtxRangeEnd(nvtx0);
		cudaThreadSynchronize();
		return true;
	} // GTile::compute_dft2()
	#endif


	// ////
	// Testing functions
	// ////

	__host__ void GTile::print_f_mat(unsigned int buff_i) {
		cudaMemcpy(complex_buff_h_, f_mat_[buff_i], tile_size_ * tile_size_ * sizeof(cucomplex_t), cudaMemcpyDeviceToHost);
		for(int i = 0; i < tile_size_; ++ i) {
			for(int j = 0; j < tile_size_; ++ j) {
				std::cout << complex_buff_h_[tile_size_ * i + j].x << "+" << complex_buff_h_[tile_size_ * i + j].y << " ";
			} // for
			std::cout << std::endl;
		} // for 
	} // GTile::print_f_mat()


	__host__ void GTile::print_modf_mat(unsigned int buff_i) {
		cudaMemcpy(real_buff_h_, mod_f_mat_[buff_i], tile_size_ * tile_size_ * sizeof(real_t), cudaMemcpyDeviceToHost);
		for(int i = 0; i < tile_size_; ++ i) {
			for(int j = 0; j < tile_size_; ++ j) {
				std::cout << real_buff_h_[tile_size_ * i + j] << " ";
			} // for
			std::cout << std::endl;
		} // for 
	} // GTile::print_f_mat()


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
			outmat[index] = temp.x * temp.x + temp.y * temp.y;
			//outmat[index] = mask[index] * (temp.x * temp.x + temp.y * temp.y);
		} // if
	} // compute_mod_mat_kernel()


	__global__ void normalize_mod_mat_kernel(real_t* mod_mat, unsigned int size,
												real_t min_val, real_t max_val) {
		unsigned int i_x = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int i_y = blockDim.y * blockIdx.y + threadIdx.y;
		if(i_x < size && i_y < size) {
			unsigned int index = size * i_y + i_x;
			mod_mat[index] = (mod_mat[index] - min_val) / (max_val - min_val);
		} // if
	} // normalize_mod_mat_kernel()


	// this is not used
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
			unsigned int swap_i_x = (i_x + (size >> 1)) % size;
			unsigned int swap_i_y = (i_y + (size >> 1)) % size;
			unsigned int index = size * i_y + i_x;
			unsigned int swap_index = size * swap_i_y + swap_i_x;
			//real_t temp = pattern[swap_index] - mod_mat[index] * c_factor;
			//real_t temp = pattern[swap_index] - 2.5 * mod_mat[index];
			real_t temp = pattern[swap_index] - mod_mat[index];
			out[index] = temp * temp;
		} // if
	} // compute_chi2_kernel()


	#ifdef USE_DFT
	// base kernel - no shared mem
	__global__ void compute_dft2_kernel(cucomplex_t* vandermonde, unsigned int size,
										unsigned int old_row, unsigned int old_col,
										unsigned int new_row, unsigned int new_col,
										unsigned int num_particles,
										cucomplex_t* fin, cucomplex_t* fout) {
		unsigned int i_x = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int i_y = blockDim.y * blockIdx.y + threadIdx.y;
		if(i_x < size && i_y < size) {
			//unsigned int index = size * i_y + i_x;
			unsigned int index_t = size * i_x + i_y;	// transpose
			cucomplex_t new_temp = complex_mul(vandermonde[size * i_y + new_col],
												vandermonde[size * new_row + i_x]);
			cucomplex_t old_temp = complex_mul(vandermonde[size * i_y + old_col],
												vandermonde[size * old_row + i_x]);
			cucomplex_t dft_temp = complex_sub(new_temp, old_temp);
			//cucomplex_t dft_temp = complex_div((complex_sub(new_temp, old_temp)), (real_t)num_particles);
			fout[index_t] = complex_add(dft_temp, fin[index_t]);
		} // if
	} // compute_dft2_kernel()


	// shared mem - naive
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
		if(threadIdx.y == 0 && i_y < size) {
			vander_old_col[threadIdx.x] = vandermonde[old_y];
			vander_new_col[threadIdx.x] = vandermonde[new_y];
		} // if

		__syncthreads();	// make sure all data is available

		unsigned int index = size * i_y + i_x;
		unsigned int index_t = size * i_x + i_y;	// transpose
		if(i_x < size && i_y < size) {
			cucomplex_t new_temp = complex_mul(vander_new_col[threadIdx.y],
								vander_new_row[threadIdx.x]);
			cucomplex_t old_temp = complex_mul(vander_old_col[threadIdx.y],
								vander_old_row[threadIdx.x]);
			cucomplex_t dft_temp = complex_div((complex_sub(new_temp, old_temp)),
								(real_t)num_particles);
			fout[index_t] = complex_add(dft_temp, fin[index]);
		} // if
	} // compute_dft2_kernel_shared()


	// shared mem - rowwise to avoid bank conflicts -- verrrry slooooow
	__global__ void compute_dft2_kernel_shared_opt(cucomplex_t* vandermonde, unsigned int size,
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
		unsigned int index_t = size * i_x + i_y;	// transpose
		if(i_x < size && i_y < size) {
			for(int i = 0; i < blockDim.x; ++ i) {		// to remove shared mem bank conflicts
				if(threadIdx.x == i) {
					// y threads read serial memory
					// x threads use one broadcast
					cucomplex_t new_temp = complex_mul(vander_new_col[threadIdx.y],
										vander_new_row[threadIdx.x]);
					cucomplex_t old_temp = complex_mul(vander_old_col[threadIdx.y],
										vander_old_row[threadIdx.x]);
					cucomplex_t dft_temp = complex_div((complex_sub(new_temp, old_temp)),
										(real_t)num_particles);
					fout[index_t] = complex_add(dft_temp, fin[index]);
				} // if
			} // for
		} // if
	} // compute_dft2_kernel_shared()


	// shared mem - padding or 8 thread groups at a time
	__global__ void compute_dft2_kernel_shared_opt2(cucomplex_t* vandermonde, unsigned int size,
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

		// assuming blocks are 16x16
		unsigned int padx = (threadIdx.x < 8) ? 0 : 1;
		unsigned int pady = (threadIdx.y < 8) ? 0 : 1;

		//__shared__ cucomplex_t vander_old_row[CUDA_BLOCK_SIZE_X_];
		//__shared__ cucomplex_t vander_new_row[CUDA_BLOCK_SIZE_X_];
		//__shared__ cucomplex_t vander_old_col[CUDA_BLOCK_SIZE_Y_];
		//__shared__ cucomplex_t vander_new_col[CUDA_BLOCK_SIZE_Y_];
		__shared__ real_t vander_old_row[2 * CUDA_BLOCK_SIZE_X_ + 1];
		__shared__ real_t vander_new_row[2 * CUDA_BLOCK_SIZE_X_ + 1];
		__shared__ real_t vander_old_col[2 * CUDA_BLOCK_SIZE_Y_ + 1];
		__shared__ real_t vander_new_col[2 * CUDA_BLOCK_SIZE_Y_ + 1];
		/*__shared__ real_t vander[4 * CUDA_BLOCK_SIZE_X_ + 4 * CUDA_BLOCK_SIZE_Y_ + 8];
		real_t *vander_old_row = vander;
		real_t *vander_new_row = vander_old_row + 2 * CUDA_BLOCK_SIZE_X_ + 2;
		real_t *vander_old_col = vander_new_row + 2 * CUDA_BLOCK_SIZE_X_ + 2;
		real_t *vander_new_col = vander_old_col + 2 * CUDA_BLOCK_SIZE_Y_ + 2;*/

		unsigned int t_i_x = 2 * threadIdx.x + padx;
		unsigned int t_i_y = 2 * threadIdx.y + pady;

		// first row of threads load both rows
		if(threadIdx.y == 0 && i_x < size) {
			cucomplex_t temp1 = vandermonde[old_x];
			cucomplex_t temp2 = vandermonde[new_x];
			vander_old_row[t_i_x] = temp1.x;
			vander_old_row[t_i_x + 1] = temp1.y;
			vander_new_row[t_i_x] = temp2.x;
			vander_new_row[t_i_x + 1] = temp2.y;
		} // if
		// first col of threads load both cols
		if(threadIdx.x == 0 && i_y < size) {
			cucomplex_t temp1 = vandermonde[old_y];
			cucomplex_t temp2 = vandermonde[new_y];
			vander_old_col[t_i_y] = temp1.x;
			vander_old_col[t_i_y + 1] = temp1.y;
			vander_new_col[t_i_y] = temp2.x;
			vander_new_col[t_i_y + 1] = temp2.y;
		} // if

		__syncthreads();	// make sure all data is available

		//unsigned int r_iter = blockDim.x >> 3;
		//unsigned int c_iter = blockDim.y >> 3;
		unsigned int index = size * i_y + i_x;
		unsigned int index_t = size * i_x + i_y;	// transpose
		if(i_x < size && i_y < size) {
			//for(int r = 0; r < r_iter; ++ r) {
			//	if(threadIdx.x >= r << 3 && threadIdx.x < (r + 1) << 3) {
					cucomplex_t new_row = make_cuComplex(vander_new_row[t_i_x],
															vander_new_row[t_i_x + 1]);
					cucomplex_t old_row = make_cuComplex(vander_old_row[t_i_x],
															vander_old_row[t_i_x + 1]);
					//for(int c = 0; c < c_iter; ++ c) {
					//	if(threadIdx.y >= c << 3 && threadIdx.y < (c + 1) << 3) {
							cucomplex_t new_col = make_cuComplex(vander_new_col[t_i_y],
																	vander_new_col[t_i_y + 1]);
							cucomplex_t old_col = make_cuComplex(vander_old_col[t_i_y],
																	vander_old_col[t_i_y + 1]);
							cucomplex_t new_temp = complex_mul(new_col, new_row);
							cucomplex_t old_temp = complex_mul(old_col, old_row);
							cucomplex_t dft_temp = complex_div((complex_sub(new_temp, old_temp)),
																(real_t) num_particles);
							fout[index_t] = complex_add(dft_temp, fin[index]);
					//	} // if c
					//} // for c
			//	} // if r
			//} // for r
		} // if
	} // compute_dft2_kernel_shared()


	// with subtiling (no padding)
	__global__ void compute_dft2_kernel_shared_opt3(cucomplex_t* vandermonde, unsigned int size,
							unsigned int old_row, unsigned int old_col,
							unsigned int new_row, unsigned int new_col,
							unsigned int num_particles,
							cucomplex_t* fin, cucomplex_t* fout) {
		// TODO: try subtiling also
		// TODO: try dynamic shared memory
		// TODO: try shared mem for output to coalesce writes
		unsigned int i_x = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int i_y = (blockDim.y * CUDA_DFT2_SUBTILES_) * blockIdx.y + threadIdx.y;

		unsigned int old_x = size * old_row + i_x;
		unsigned int old_y = size * i_y + old_col;
		unsigned int new_x = size * new_row + i_x;
		unsigned int new_y = size * i_y + new_col;

		__shared__ cucomplex_t vander_old_row[CUDA_BLOCK_SIZE_X_];
		__shared__ cucomplex_t vander_new_row[CUDA_BLOCK_SIZE_X_];
		__shared__ cucomplex_t vander_old_col[CUDA_BLOCK_SIZE_Y_];
		__shared__ cucomplex_t vander_new_col[CUDA_BLOCK_SIZE_Y_];

		// first row of threads load both rows
		if(threadIdx.y == 0 && i_x < size) {
			vander_old_row[threadIdx.x] = vandermonde[old_x];
			vander_new_row[threadIdx.x] = vandermonde[new_x];
		} // if
		for(int s = 0; s < CUDA_DFT2_SUBTILES_; ++ s) {
			// col of threads load both cols of surrent subtile
			old_y = size * i_y + old_col + size * blockDim.y * s;
			new_y = size * i_y + new_col + size * blockDim.y * s;
			if(threadIdx.x == 0 && i_y < size) {
				vander_old_col[threadIdx.y] = vandermonde[old_y];
				vander_new_col[threadIdx.y] = vandermonde[new_y];
			} // if

			__syncthreads();	// make sure all data is available

			unsigned int index = size * i_y + size * blockDim.y * s + i_x;
			unsigned int index_t = size * i_y + size * blockDim.y * s + i_x;	// transpose
			if(i_x < size && i_y < size) {
				cucomplex_t new_temp = complex_mul(vander_new_col[threadIdx.y],
									vander_new_row[threadIdx.x]);
				cucomplex_t old_temp = complex_mul(vander_old_col[threadIdx.y],
									vander_old_row[threadIdx.x]);
				cucomplex_t dft_temp = complex_div((complex_sub(new_temp, old_temp)),
									(real_t)num_particles);
				fout[index_t] = complex_add(dft_temp, fin[index]);
			} // if
		}
	} // compute_dft2_kernel_shared_opt3()


	// subtiling plus padding
	__global__ void compute_dft2_kernel_shared_opt4(cucomplex_t* vandermonde, unsigned int size,
							unsigned int old_row, unsigned int old_col,
							unsigned int new_row, unsigned int new_col,
							unsigned int num_particles,
							cucomplex_t* fin, cucomplex_t* fout) {
		// TODO: try dynamic shared memory
		// TODO: try shared mem for output to coalesce writes
		unsigned int i_x = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int i_y = (blockDim.y * CUDA_DFT2_SUBTILES_) * blockIdx.y + threadIdx.y;

		unsigned int old_x = size * old_row + i_x;
		unsigned int old_y = size * i_y + old_col;
		unsigned int new_x = size * new_row + i_x;
		unsigned int new_y = size * i_y + new_col;

		unsigned int padx = (threadIdx.x < 8) ? 0 : 1;
		unsigned int pady = (threadIdx.y < 8) ? 0 : 1;

		__shared__ real_t vander_old_row[2 * CUDA_BLOCK_SIZE_X_ + 1];
		__shared__ real_t vander_new_row[2 * CUDA_BLOCK_SIZE_X_ + 1];
		__shared__ real_t vander_old_col[2 * CUDA_BLOCK_SIZE_Y_ + 1];
		__shared__ real_t vander_new_col[2 * CUDA_BLOCK_SIZE_Y_ + 1];

		unsigned int t_i_x = 2 * threadIdx.x + padx;
		unsigned int t_i_y = 2 * threadIdx.y + pady;

		// first row of threads load both rows
		if(threadIdx.y == 0 && i_x < size) {
			cucomplex_t temp1 = vandermonde[old_x];
			cucomplex_t temp2 = vandermonde[new_x];
			vander_old_row[t_i_x] = temp1.x;
			vander_old_row[t_i_x + 1] = temp1.y;
			vander_new_row[t_i_x] = temp2.x;
			vander_new_row[t_i_x + 1] = temp2.y;
		} // if
		for(int s = 0; s < CUDA_DFT2_SUBTILES_; ++ s) {
			// col of threads load both cols of surrent subtile
			old_y = size * i_y + old_col + size * blockDim.y * s;
			new_y = size * i_y + new_col + size * blockDim.y * s;
			if(threadIdx.x == 0 && i_y < size) {
				cucomplex_t temp1 = vandermonde[old_y];
				cucomplex_t temp2 = vandermonde[new_y];
				vander_old_col[t_i_y] = temp1.x;
				vander_old_col[t_i_y + 1] = temp1.y;
				vander_new_col[t_i_y] = temp2.x;
				vander_new_col[t_i_y + 1] = temp2.y;
			} // if

			__syncthreads();	// make sure all data is available

			unsigned int index = size * i_y + size * blockDim.y * s + i_x;
			unsigned int index_t = size * i_x + size * blockDim.x * s + i_y; // for transpose
			if(i_x < size && i_y < size) {
				cucomplex_t new_row = make_cuComplex(vander_new_row[t_i_x], vander_new_row[t_i_x + 1]);
				cucomplex_t old_row = make_cuComplex(vander_old_row[t_i_x], vander_old_row[t_i_x + 1]);
				cucomplex_t new_col = make_cuComplex(vander_new_col[t_i_y], vander_new_col[t_i_y + 1]);
				cucomplex_t old_col = make_cuComplex(vander_old_col[t_i_y], vander_old_col[t_i_y + 1]);
				cucomplex_t new_temp = complex_mul(new_col, new_row);
				cucomplex_t old_temp = complex_mul(old_col, old_row);
				cucomplex_t dft_temp = complex_div((complex_sub(new_temp, old_temp)), (real_t) num_particles);
				fout[index_t] = complex_add(dft_temp, fin[index]);
			} // if
		} // for
	} // compute_dft2_kernel_shared_opt4()


	__global__ void compute_dft2_kernel_shared_test(cucomplex_t* vandermonde, unsigned int size,
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
		const unsigned int mat_size = CUDA_BLOCK_SIZE_X_ * CUDA_BLOCK_SIZE_Y_;
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
		} // if

		__syncthreads();	// make sure all data is available
		unsigned int num_iter = mat_size >> 3;	// to avoid bank conflicts
												// this is the case with cucomplex_t = double complex
		unsigned int out_index = size * i_y + i_x;
		if(i_x < size && i_y < size) {
			for(int i = 0; i < num_iter; ++ i) {
				if(in_index > (i - 1) << 3 && in_index < i << 3) {
					cucomplex_t new_temp = complex_mul(vander_new_col[in_index],
										vander_new_row[in_index]);
					cucomplex_t old_temp = complex_mul(vander_old_col[in_index],
										vander_old_row[in_index]);
					cucomplex_t dft_temp = complex_div((complex_sub(new_temp, old_temp)),
										(real_t)num_particles);
					fout[out_index] = complex_add(dft_temp, fin[out_index]);
				} // if
			} // for
		} // if
	} // compute_dft2_kernel_shared()

	#endif // USE_DFT

} // namespace hir
