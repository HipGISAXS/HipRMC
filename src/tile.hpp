/***
  *  Project: HipRMC - High Performance Reverse Monte Carlo Simulations
  *
  *  File: tile.hpp
  *  Created: Jan 25, 2013
  *  Modified: Sat 02 Feb 2013 01:42:09 PM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef __TILE_HPP__
#define __TILE_HPP__

#include <vector>
#include <complex>
#include <random>
#include <omp.h>
#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <fftw3.h>
#include <opencv2/opencv.hpp>
#include <woo/matrix/matrix.hpp>

#include "temp.hpp"

namespace hir {

	template <typename real_t, typename complex_t, typename cucomplex_t>
	class Tile {
		private:
			// following define a tile
			unsigned int size_;									// num rows = num cols = size
			woo::Matrix2D<real_t> a_mat_;						// A
			std::vector<woo::Matrix2D<complex_t> > f_mat_;		// F buffers
			std::vector<woo::Matrix2D<real_t> > mod_f_mat_;		// auto_F buffers
			std::vector<unsigned int> indices_;					// NOTE: the first num_particles_ entries
																// in indices_ are 'filled', rest are 'empty'
			unsigned int f_mat_i_;								// current f buffer index
			unsigned int mod_f_mat_i_;							// current mod_f buffer index
			real_t loading_factor_;								// loading factor
			unsigned int num_particles_;						// number of particles (duh!)
			double model_norm_;									// norm of current model
			double c_factor_;									// c factor

			// buffers for device
			cucomplex_t* a_mat_d_;								// not really needed
			cucomplex_t* f_mat_d_;

			// following are used during simulation
			woo::Matrix2D<complex_t> dft_mat_;
			double prev_chi2_;

			// indices produced on virtually moving a particle
			unsigned int old_pos_;
			unsigned int new_pos_;
			unsigned int old_index_;
			unsigned int new_index_;

			// for random number generation
			std::mt19937_64 ms_rand_gen_;

			// functions
			bool compute_fft_mat_cpu();									// cpu
			bool compute_fft_mat_cuda();								// gpu
			bool execute_fftw(fftw_complex*, fftw_complex*);			// cpu
			bool execute_cufft(cuFloatComplex*, cuFloatComplex*);		// gpu
			bool execute_cufft(cuDoubleComplex*, cuDoubleComplex*);		// gpu
			bool compute_mod_mat(const woo::Matrix2D<complex_t>&);		// cpu
			bool compute_model_norm(const woo::Matrix2D<real_t>&);		// cpu
			double compute_chi2(const woo::Matrix2D<real_t>&, const woo::Matrix2D<real_t>&, real_t);	// cpu
			bool virtual_move_random_particle();						// cpu
			bool move_particle(double, real_t);							// cpu
			bool compute_dft2(woo::Matrix2D<complex_t>&, woo::Matrix2D<complex_t>&);	// cpu
			bool update_fft_mat(woo::Matrix2D<complex_t>&, woo::Matrix2D<complex_t>&,
								woo::Matrix2D<complex_t>&);								// cpu
			bool mask_mat(const unsigned int*&, const woo::Matrix2D<real_t>&);			// cpu

		public:
			Tile(unsigned int, unsigned int, const std::vector<unsigned int>&);
			Tile(const Tile&);
			~Tile();

			// initialize with raw data
			bool init(real_t, real_t, const woo::Matrix2D<real_t>&, const unsigned int*);
			bool simulate_step(const woo::Matrix2D<real_t>&, woo::Matrix2D<complex_t>&,
								const unsigned int*, real_t, real_t);
			bool update_model(const woo::Matrix2D<real_t>&, real_t);
			bool finalize_result(double&, woo::Matrix2D<real_t>&);

			// return a random number in (0,1)
			real_t ms_rand_01() {
				return ((real_t) (ms_rand_gen_() - ms_rand_gen_.min()) /
									(ms_rand_gen_.max() - ms_rand_gen_.min()));
			} // ms_rand_01()

	}; // class Tile


	template <typename real_t, typename complex_t, typename cucomplex_t>
	Tile<real_t, complex_t, cucomplex_t>::Tile(unsigned int rows, unsigned int cols,
												const std::vector<unsigned int>& indices) :
		a_mat_(rows, cols),
		f_mat_i_(0),
		mod_f_mat_i_(0),
		indices_(indices),
		dft_mat_(rows, cols) {

		size_ = std::max(rows, cols);
		// two buffers each
		f_mat_.push_back(woo::Matrix2D<complex_t>(size_, size_));
		f_mat_.push_back(woo::Matrix2D<complex_t>(size_, size_));
		mod_f_mat_.push_back(woo::Matrix2D<real_t>(size_, size_));
		mod_f_mat_.push_back(woo::Matrix2D<real_t>(size_, size_));
		cudaMalloc((void**) &a_mat_d_, size_ * size_ * sizeof(cucomplex_t));
		cudaMalloc((void**) &f_mat_d_, size_ * size_ * sizeof(cucomplex_t));
	} // Tile::Tile()


	// copy constructor
	template <typename real_t, typename complex_t, typename cucomplex_t>
	Tile<real_t, complex_t, cucomplex_t>::Tile(const Tile<real_t, complex_t, cucomplex_t>& tile):
		size_(tile.size_),
		a_mat_(tile.a_mat_),
		f_mat_(tile.f_mat_),
		mod_f_mat_(tile.mod_f_mat_),
		indices_(tile.indices_),
		f_mat_i_(tile.f_mat_i_),
		mod_f_mat_i_(tile.mod_f_mat_i_),
		loading_factor_(tile.loading_factor_),
		num_particles_(tile.num_particles_),
		model_norm_(tile.model_norm_),
		c_factor_(tile.c_factor_),
		a_mat_d_(NULL),
		f_mat_d_(NULL),
		dft_mat_(tile.dft_mat_),
		prev_chi2_(tile.prev_chi2_),
		old_pos_(tile.old_pos_),
		new_pos_(tile.new_pos_),
		old_index_(tile.old_index_),
		new_index_(tile.new_index_) {
		cudaMalloc((void**) &a_mat_d_, size_ * size_ * sizeof(cucomplex_t));
		cudaMalloc((void**) &f_mat_d_, size_ * size_ * sizeof(cucomplex_t));
	} // Tile::Tile()

	template <typename real_t, typename complex_t, typename cucomplex_t>
	Tile<real_t, complex_t, cucomplex_t>::~Tile() {
		cudaFree(a_mat_d_);
		cudaFree(f_mat_d_);
	} // Tile::~Tile()


	// initialize with raw data
	template <typename real_t, typename complex_t, typename cucomplex_t>
	bool Tile<real_t, complex_t, cucomplex_t>::init(real_t loading, real_t base_norm,
													const woo::Matrix2D<real_t>& pattern,
													const unsigned int* mask) {
		woo::BoostChronoTimer mytimer;
		//srand(time(NULL));
		unsigned seed = time(NULL); //std::chrono::system_clock::now().time_since_epoch().count();
		ms_rand_gen_.seed(seed);

		loading_factor_ = loading;
		unsigned int cols = a_mat_.num_cols(), rows = a_mat_.num_rows();
		num_particles_ = ceil(loading * rows * cols);
		// NOTE: the first num_particles_ entries in indices_ are filled, rest are empty
		std::cout << "++ num_particles: " << num_particles_ << std::endl;

		// fill a_mat_ with particles
		mytimer.start();
		a_mat_.fill(0.0);
		for(unsigned int i = 0; i < num_particles_; ++ i) {
			unsigned int x = indices_[i] / cols;
			unsigned int y = indices_[i] % cols;
			a_mat_(x, y) = 1.0;
		} // for
		mytimer.stop();
		std::cout << "**** A fill time: " << mytimer.elapsed_msec() << " ms." << std::endl;
		//print_matrix("a_mat", a_mat_.data(), rows, cols);

		// compute fft of a_mat_ into fft_mat_ and other stuff
		mytimer.start();
		compute_fft_mat_cuda();
		//compute_fft_mat_cpu();
		mytimer.stop();
		std::cout << "**** FFT time: " << mytimer.elapsed_msec() << " ms." << std::endl;
		mytimer.start();
		compute_mod_mat(f_mat_[f_mat_i_]);
		mytimer.stop();
		std::cout << "**** mod F time: " << mytimer.elapsed_msec() << " ms." << std::endl;
		mytimer.start();
		mask_mat(mask, mod_f_mat_[1 - mod_f_mat_i_]);
		mytimer.stop();
		std::cout << "**** Mask time: " << mytimer.elapsed_msec() << " ms." << std::endl;
		mod_f_mat_[mod_f_mat_i_].populate(mod_f_mat_[1 - mod_f_mat_i_].data());
		mytimer.start();
		compute_model_norm(mod_f_mat_[1 - mod_f_mat_i_]);
		mytimer.stop();
		std::cout << "**** model norm time: " << mytimer.elapsed_msec() << " ms." << std::endl;
		c_factor_ = base_norm / model_norm_;
		//std::cout << "++++ c_factor: " << c_factor_ << std::endl;
		mytimer.start();
		prev_chi2_ = compute_chi2(pattern, mod_f_mat_[1 - mod_f_mat_i_], c_factor_);
		mytimer.stop();
		std::cout << "**** chi2 time: " << mytimer.elapsed_msec() << " ms." << std::endl;
		return true;
	} // Tile::init()


	template<typename real_t, typename complex_t, typename cucomplex_t>
	bool Tile<real_t, complex_t, cucomplex_t>::simulate_step(const woo::Matrix2D<real_t>& pattern,
												woo::Matrix2D<complex_t>& vandermonde,
												const unsigned int* mask,
												real_t tstar, real_t base_norm) {
		//std::cout << "++ simulate_step" << std::endl;
		// do all computations in scratch buffers
		unsigned int f_scratch_i = 1 - f_mat_i_;
		unsigned int mod_f_scratch_i = 1 - mod_f_mat_i_;

		virtual_move_random_particle();
		// merge the following two: ...
		compute_dft2(vandermonde, dft_mat_);
		update_fft_mat(dft_mat_, f_mat_[f_mat_i_], f_mat_[f_scratch_i]);
		//print_cmatrix("f_mat_[f_scratch_i]", f_mat_[f_scratch_i].data(), size_, size_);
		compute_mod_mat(f_mat_[f_scratch_i]);
		mask_mat(mask, mod_f_mat_[mod_f_scratch_i]);
		// this should be here i think ...
		compute_model_norm(mod_f_mat_[mod_f_scratch_i]);
		double new_c_factor = base_norm / model_norm_;
		double new_chi2 = compute_chi2(pattern, mod_f_mat_[mod_f_scratch_i], new_c_factor);
		double diff_chi2 = prev_chi2_ - new_chi2;
		std::cout << "++++ chi2 diff:\t" << prev_chi2_ << "\t" << new_chi2 << "\t" << diff_chi2
					<< "\t c_factor: " << new_c_factor << std::endl;

		bool accept = false;
		if(new_chi2 < prev_chi2_) accept = true;
		else {
			real_t p = exp(diff_chi2 / tstar);
			real_t prand = ms_rand_01();
			//std::cout << "++++ p: " << p << ", prand: " << prand << std::endl;
			if(prand < p) accept = true;
		} // if-else
		if(accept) {	// accept the move
			std::cout << std::endl << "++++ accepting move..., chi2: " << new_chi2 << ", prev: " << prev_chi2_
						<< ", old cf: " << c_factor_ << std::endl;
			// update to newly computed stuff
			// make scratch as current
			move_particle(new_chi2, base_norm);
			compute_model_norm(mod_f_mat_[mod_f_mat_i_]);
			c_factor_ = base_norm / model_norm_;
			//std::cout << "++++ accepted, new cf: " << c_factor_ << std::endl;
		} // if

		/*real_t p = std::min(1.0, exp(diff_chi2 / tstar));
		//real_t prand = (real_t)rand() / RAND_MAX;
		real_t prand = (real_t) (ms_rand_gen_() - ms_rand_gen_.min()) /
								(ms_rand_gen_.max() - ms_rand_gen_.min());
		std::cout << "++++ p: " << p << ", prand: " << prand << std::endl;
		if(prand <= p) {	// accept the move
			std::cout << "++++ accepting move..." << std::endl;
			// update to newly computed stuff
			// make scratch as current
			move_particle();
		} // if*/

		return true;
	} // Tile::simulate_step()


	template<typename real_t, typename complex_t, typename cucomplex_t>
	bool Tile<real_t, complex_t, cucomplex_t>::update_model(const woo::Matrix2D<real_t>& pattern,
															real_t base_norm) {
		//std::cout << "++ update_model" << std::endl;
//		compute_model_norm(mod_f_mat_[mod_f_mat_i_]);
//		c_factor_ = base_norm / model_norm_;
		//std::cout << "****** hehe: prev_chi2: " << prev_chi2_;
//		prev_chi2_ = compute_chi2(pattern, mod_f_mat_[mod_f_mat_i_], c_factor_);
		//std::cout << ", new prev_chi2: " << prev_chi2_ << std::endl;
		return true;
	} // Tile::update_model()


	template<typename real_t, typename complex_t, typename cucomplex_t>
	bool Tile<real_t, complex_t, cucomplex_t>::compute_fft_mat_cpu() {
		std::cout << "++ compute_fft_mat_cpu" << std::endl;

		unsigned int size2 = size_ * size_;
		fftw_complex* mat_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size2);
		fftw_complex* mat_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size2);
		real_t* orig_a_mat = a_mat_.data();
		for(int i = 0; i < size2; ++ i) {
			mat_in[i][0] = orig_a_mat[i];
			mat_in[i][1] = 0.0;
			mat_out[i][0] = 0.0;
			mat_out[i][1] = 0.0;
		} // for
		//print_fftwcmatrix("mat_in", mat_in, size_, size_);
		// execute fft
		execute_fftw(mat_in, mat_out);
		//#pragma omp parallel
		//{
#ifdef _OPENMP
		//if(omp_get_thread_num() == 0)
		//	std::cout << "[" << omp_get_num_threads() << " threads] ... " << std::flush;
#endif
		//#pragma omp for
		for(unsigned int i = 0; i < size_; ++ i) {
			for(unsigned int j = 0; j < size_; ++ j) {
				f_mat_[f_mat_i_](i, j) = complex_t(mat_out[size_ * i + j][0], mat_out[size_ * i + j][1]) /
											(real_t)num_particles_;
			}
		} // for
		//} // omp parallel

		//print_fftwcmatrix("mat_out", mat_out, size_, size_);
		//print_cmatrix("f_mat_[f_mat_i_]", f_mat_[f_mat_i_].data(), size_, size_);

		fftw_free(mat_out);
		fftw_free(mat_in);
		return true;
	} // Tile::compute_fft_mat_cpu()


	template<typename real_t, typename complex_t, typename cucomplex_t>
	bool Tile<real_t, complex_t, cucomplex_t>::execute_fftw(fftw_complex* input,
															fftw_complex* output) {
		// create fft plan
		fftw_plan plan;
		plan = fftw_plan_dft_2d(size_, size_, input, output, FFTW_FORWARD, FFTW_ESTIMATE);
		fftw_execute(plan);
		// destroy fft plan
		fftw_destroy_plan(plan);
		return true;
	} // Tile::execute_cufft()


	template<typename real_t, typename complex_t, typename cucomplex_t>
	bool Tile<real_t, complex_t, cucomplex_t>::compute_fft_mat_cuda() {
		std::cout << "++ compute_fft_mat_cuda" << std::endl;

		unsigned int size2 = size_ * size_;
		cucomplex_t* temp_mat = new (std::nothrow) cucomplex_t[size2];
		real_t* orig_a_mat = a_mat_.data();
		for(int i = 0; i < size2; ++ i) {
			temp_mat[i].x = orig_a_mat[i];
			temp_mat[i].y = 0.0;
		} // for
		//print_cucmatrix("temp_mat", temp_mat, size_, size_);
		// copy data to device
		cudaMemcpy(a_mat_d_, temp_mat, size2 * sizeof(cucomplex_t), cudaMemcpyHostToDevice);
		// execute fft
		execute_cufft(a_mat_d_, f_mat_d_);
		// copy data to host, reuse the temp_mat buffer
		//cucomplex_t* temp_mat2 = new (std::nothrow) cucomplex_t[size2];
		cudaMemcpy(temp_mat, f_mat_d_, size2 * sizeof(cucomplex_t), cudaMemcpyDeviceToHost);
		#pragma omp parallel for
		for(unsigned int i = 0; i < size_; ++ i) {
			for(unsigned int j = 0; j < size_; ++ j) {
				f_mat_[f_mat_i_](i, j) = complex_t(temp_mat[size_ * i + j].x, temp_mat[size_ * i + j].y) /
											(real_t)num_particles_;
			}
		} // for

		//print_cucmatrix("temp_mat", temp_mat, size_, size_);
		//print_cmatrix("f_mat_[f_mat_i_]", f_mat_[f_mat_i_].data(), size_, size_);

		//delete[] temp_mat2;
		delete[] temp_mat;
		return true;
	} // Tile::compute_fft_mat_cuda()


	template<typename real_t, typename complex_t, typename cucomplex_t>
	bool Tile<real_t, complex_t, cucomplex_t>::execute_cufft(cuFloatComplex* input,
															cuFloatComplex* output) {
		// create fft plan
		cufftHandle plan;
		cufftResult res;
		res = cufftPlan2d(&plan, size_, size_, CUFFT_C2C);
		if(res != CUFFT_SUCCESS) {
			std::cerr << "error: " << res << ": fft plan could not be created" << std::endl;
			return false;
		} // if
		res = cufftExecC2C(plan, input, output, CUFFT_FORWARD);
		if(res != CUFFT_SUCCESS) {
			std::cerr << "error: " << res << ": fft could not be executed" << std::endl;
			return false;
		} // if
		cudaThreadSynchronize();
		// destroy fft plan
		cufftDestroy(plan);
		return true;
	} // Tile::execute_cufft()


	template<typename real_t, typename complex_t, typename cucomplex_t>
	bool Tile<real_t, complex_t, cucomplex_t>::execute_cufft(cuDoubleComplex* input,
															cuDoubleComplex* output) {
		// create fft plan
		cufftHandle plan;
		cufftPlan2d(&plan, size_, size_, CUFFT_Z2Z);
		cufftExecZ2Z(plan, input, output, CUFFT_FORWARD);
		cudaThreadSynchronize();
		// destroy fft plan
		cufftDestroy(plan);
		return true;
	} // Tile::execute_cufft()


	template<typename real_t, typename complex_t, typename cucomplex_t>
	bool Tile<real_t, complex_t, cucomplex_t>::compute_mod_mat(const woo::Matrix2D<complex_t>& mat) {
		//std::cout << "++ compute_mod_mat" << std::endl;
		#pragma omp parallel for collapse(2)
		for(unsigned int i = 0; i < size_; ++ i) {
			for(unsigned int j = 0; j < size_; ++ j) {
				complex_t temp_f = mat(i, j);
				mod_f_mat_[1 - mod_f_mat_i_](i, j) = temp_f.real() * temp_f.real() +
													temp_f.imag() * temp_f.imag();
			} // for
		} // for
		//print_matrix("mod_f_mat_[1 - mod_f_mat_i_]", mod_f_mat_[1 - mod_f_mat_i_].data(), size_, size_);
		return true;
	} // Tile::compute_mod_mat()


	template<typename real_t, typename complex_t, typename cucomplex_t>
	bool Tile<real_t, complex_t, cucomplex_t>::compute_model_norm(const woo::Matrix2D<real_t>& mod_mat) {
		//std::cout << "++ compute_model_norm" << std::endl;
		double model_norm = 0.0;
		#pragma omp parallel shared(model_norm)
		{
			/*#pragma omp for	collapse(2)
			for(unsigned int i = 0; i < size_ / 2; ++ i) {
				for(unsigned int j = 0; j < size_ / 2; ++ j) {
					#pragma omp critical
					model_norm += mod_mat(i, j) * (j + 1);			// what is the Y matrix anyway ???
				} // for
			} // for*/
			#pragma omp for collapse(2) reduction(+:model_norm)
			for(unsigned int i = 0; i < size_ / 2; ++ i) {
				for(unsigned int j = 0; j < size_ / 2; ++ j) {
					model_norm += mod_mat(i, j) * (j + 1);			// what is the Y matrix anyway ???
				} // for
			} // for
		}
		model_norm_ = model_norm;
		//std::cout << "++++ model_norm: " << model_norm_ << std::endl;
		return true;
	} // Tile::compute_model_norm()


	template<typename real_t, typename complex_t, typename cucomplex_t>
	double Tile<real_t, complex_t, cucomplex_t>::compute_chi2(const woo::Matrix2D<real_t>& pattern,
															const woo::Matrix2D<real_t>& mod_f,
															real_t c_factor) {
//		std::cout << "++ compute_chi2" << std::endl;
		double chi2 = 0.0;
		#pragma omp parallel for collapse(2) reduction(+:chi2)
		for(unsigned int i = 0; i < size_; ++ i) {
			for(unsigned int j = 0; j < size_; ++ j) {
				real_t temp = pattern(i, j) - mod_f(i, j) * c_factor;
				chi2 += temp * temp;
			} // for
		} // for
//		std::cout << "++++ chi2: " << chi2 << std::endl;
		return chi2;
	} // Tile::compute_chi2()


	template<typename real_t, typename complex_t, typename cucomplex_t>
	bool Tile<real_t, complex_t, cucomplex_t>::virtual_move_random_particle() {
		//std::cout << "++ virtual_move_random_particle" << std::endl;
		//old_pos_ = floor(((real_t)rand() / RAND_MAX) * num_particles_);
		//new_pos_ = floor(((real_t)rand() / RAND_MAX) * (size_ * size_ - num_particles_)) +
		//old_pos_ = floor(((real_t) (ms_rand_gen_() - ms_rand_gen_.min()) /
		//							(ms_rand_gen_.max() - ms_rand_gen_.min())) *
		//							num_particles_);
		//new_pos_ = floor(((real_t) (ms_rand_gen_() - ms_rand_gen_.min()) /
		//							(ms_rand_gen_.max() - ms_rand_gen_.min())) *
		//							(size_ * size_ - num_particles_)) + num_particles_;
		old_pos_ = floor(ms_rand_01() * num_particles_);
		new_pos_ = floor(ms_rand_01() *	(size_ * size_ - num_particles_)) + num_particles_;
		old_index_ = indices_[old_pos_];
		new_index_ = indices_[new_pos_];
		//std::cout << "++++ old_pos,new_pos: " << old_pos_ << "," << new_pos_
		//			<< ", old_index,new_index: " << old_index_ << "," << new_index_ << std::endl;
		return true;
	} // Tile::virtual_move_random_particle()


	template<typename real_t, typename complex_t, typename cucomplex_t>
	bool Tile<real_t, complex_t, cucomplex_t>::move_particle(double new_chi2, real_t base_norm) {
		//std::cout << "++ move_particle and swap buffers" << std::endl;

		// to swap buffers
		f_mat_i_ = 1 - f_mat_i_;
		mod_f_mat_i_ = 1 - mod_f_mat_i_;
		prev_chi2_ = new_chi2;

		// to swap indices (moving the particle)
		indices_[old_pos_] = new_index_;
		indices_[new_pos_] = old_index_;

		return true;
	} // Tile::move_particle()


	template<typename real_t, typename complex_t, typename cucomplex_t>
	bool Tile<real_t, complex_t, cucomplex_t>::compute_dft2(woo::Matrix2D<complex_t> &vandermonde_mat,
															woo::Matrix2D<complex_t> &dft_mat) {
		//std::cout << "++ compute_dft2" << std::endl;
		unsigned int old_row = old_index_ / size_;
		unsigned int old_col = old_index_ % size_;
		unsigned int new_row = new_index_ / size_;
		unsigned int new_col = new_index_ % size_;
		typename woo::Matrix2D<complex_t>::row_iterator old_row_iter = vandermonde_mat.row(old_row);
		typename woo::Matrix2D<complex_t>::col_iterator old_col_iter = vandermonde_mat.column(old_col);
		typename woo::Matrix2D<complex_t>::row_iterator new_row_iter = vandermonde_mat.row(new_row);
		typename woo::Matrix2D<complex_t>::col_iterator new_col_iter = vandermonde_mat.column(new_col);
		#pragma omp parallel for collapse(2)
		for(unsigned int row = 0; row < size_; ++ row) {
			for(unsigned int col = 0; col < size_; ++ col) {
				complex_t new_temp = new_col_iter[row] * new_row_iter[col];
				complex_t old_temp = old_col_iter[row] * old_row_iter[col];
				//complex_t new_temp = vandermonde_mat(row, new_col) * vandermonde_mat(new_row, col);
				//complex_t old_temp = vandermonde_mat(row, old_col) * vandermonde_mat(old_row, col);
				dft_mat(row, col) = (new_temp - old_temp) / (real_t)num_particles_;
			} // for
		} // for
		//print_cmatrix("dft_mat", dft_mat.data(), size_, size_);
		return true;
	} // Tile::compute_dft2()


	template<typename real_t, typename complex_t, typename cucomplex_t>
	bool Tile<real_t, complex_t, cucomplex_t>::update_fft_mat(woo::Matrix2D<complex_t> &dft_mat,
																woo::Matrix2D<complex_t> &f_mat,
																woo::Matrix2D<complex_t> &new_f_mat) {
		return woo::matrix_add(dft_mat, f_mat, new_f_mat);	// matrix operation
	} // Tile::update_fft_mat()


	template<typename real_t, typename complex_t, typename cucomplex_t>
	bool Tile<real_t, complex_t, cucomplex_t>::mask_mat(const unsigned int*& mask_mat,
														const woo::Matrix2D<real_t>& mat) {
		#pragma omp parallel for collapse(2) shared(mat, mask_mat)
		for(unsigned int i = 0; i < size_; ++ i) {
			for(unsigned int j = 0; j < size_; ++ j) {
				mat(i, j) *= mask_mat[size_ * i + j];
			} // for
		} // for
		return true;
	} // Tile::mask_mat()


	template<typename real_t, typename complex_t, typename cucomplex_t>
	bool Tile<real_t, complex_t, cucomplex_t>::finalize_result(double& chi2, woo::Matrix2D<real_t>& a) {
		// populate a_mat_
		a_mat_.fill(0.0);
		#pragma omp parallel for
		for(unsigned int i = 0; i < num_particles_; ++ i) {
			unsigned int x = indices_[i] / size_;
			unsigned int y = indices_[i] % size_;
			a_mat_(x, y) = 1.0;
		} // for
		a = a_mat_;
		chi2 = prev_chi2_;
		return true;
	} // Tile::finalize()

} // namespace hir

#endif // __TILE_HPP__
