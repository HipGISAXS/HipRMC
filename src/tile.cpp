/***
  *  Project: HipRMC - High Performance Reverse Monte Carlo Simulations
  *
  *  File: tile.cpp
  *  Created: Jan 25, 2013
  *  Modified: Mon 04 Feb 2013 03:36:49 PM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

//#ifdef USE_GPU
//#include <cuda.h>
//#include <cufft.h>
//#include <cuda_runtime.h>
//#else
#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP
//#endif // USE_GPU
#include <opencv2/opencv.hpp>
#include <woo/timer/woo_boostchronotimers.hpp>

#include "tile.hpp"
#include "constants.hpp"
#include "temp.hpp"

namespace hir {

	// constructor
	Tile::Tile(unsigned int rows, unsigned int cols, const std::vector<unsigned int>& indices) :
		a_mat_(rows, cols),
		f_mat_i_(0),
		mod_f_mat_i_(0),
		indices_(indices),
		dft_mat_(rows, cols) {

		woo::BoostChronoTimer mytimer;

		size_ = std::max(rows, cols);
		// two buffers each
		mytimer.start();
		f_mat_.push_back(woo::Matrix2D<complex_t>(size_, size_));
		f_mat_.push_back(woo::Matrix2D<complex_t>(size_, size_));
		mytimer.stop();
		std::cout << "***** f_mat_: " << mytimer.elapsed_msec() << " ms." << std::endl;
		mytimer.start();
		mod_f_mat_.push_back(woo::Matrix2D<real_t>(size_, size_));
		mod_f_mat_.push_back(woo::Matrix2D<real_t>(size_, size_));
		mytimer.stop();
		std::cout << "***** mod_f_mat_: " << mytimer.elapsed_msec() << " ms." << std::endl;
#ifdef USE_GPU
		// device memory allocation takes all the time
		mytimer.start();
		unsigned int size2 = size_ * size_;
		cucomplex_buff_ = new (std::nothrow) cucomplex_t[size2];
/*		complex_buffer_h_ = new (std::nothrow) cucomplex_t[size2];
		real_buffer_h_ = new (std::nothrow) real_t[size2];
		cudaMalloc((void**) &a_mat_d_, size2 * sizeof(cucomplex_t));
		cudaMalloc((void**) &f_mat_d_[0], size2 * sizeof(cucomplex_t));
		cudaMalloc((void**) &f_mat_d_[1], size2 * sizeof(cucomplex_t));
		cudaMalloc((void**) &mod_f_mat_d_[0], size2 * sizeof(real_t));
		cudaMalloc((void**) &mod_f_mat_d_[1], size2 * sizeof(real_t));*/
		mytimer.stop();
		std::cout << "***** device mem: " << mytimer.elapsed_msec() << " ms." << std::endl;
#endif // USE_GPU
	} // Tile::Tile()


	// copy constructor
	Tile::Tile(const Tile& tile):
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
		dft_mat_(tile.dft_mat_),
		prev_chi2_(tile.prev_chi2_),
		old_pos_(tile.old_pos_),
		new_pos_(tile.new_pos_),
		old_index_(tile.old_index_),
		new_index_(tile.new_index_) {
#ifdef USE_GPU
		unsigned int size2 = size_ * size_;
		cucomplex_buff_ = new (std::nothrow) cucomplex_t[size2];
#endif // USE_GPU
	} // Tile::Tile()

	Tile::~Tile() {
#ifdef USE_GPU
		delete[] cucomplex_buff_;
#endif // USE_GPU
	} // Tile::~Tile()


	// initialize with raw data
	bool Tile::init(real_t loading, real_t base_norm, woo::Matrix2D<real_t>& pattern,
			woo::Matrix2D<complex_t>& vandermonde, const unsigned int* mask) {
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
		update_a_mat();
		//a_mat_.fill(0.0);
		//for(unsigned int i = 0; i < num_particles_; ++ i) {
		//	unsigned int x = indices_[i] / cols;
		//	unsigned int y = indices_[i] % cols;
		//	a_mat_(x, y) = 1.0;
		//} // for
		mytimer.stop();
		std::cout << "**** A fill time: " << mytimer.elapsed_msec() << " ms." << std::endl;
		//print_matrix("a_mat", a_mat_.data(), rows, cols);
#ifdef USE_GPU
		unsigned int block_x = CUDA_BLOCK_SIZE_X_;
		unsigned int block_y = CUDA_BLOCK_SIZE_Y_;
		for(unsigned int i = 0; i < size_; ++ i) {
			for(unsigned int j = 0; j < size_; ++ j) {
				complex_t temp = vandermonde(i, j);
				cucomplex_buff_[size_ * i + j].x = temp.real();
				cucomplex_buff_[size_ * i + j].y = temp.imag();
			} // for
		} // for
		gtile_.init(pattern.data(), cucomplex_buff_, a_mat_.data(), mask, size_, block_x, block_y);
#endif // USE_GPU

		// compute fft of a_mat_ into fft_mat_ and other stuff
		mytimer.start();
		compute_fft_mat();
		mytimer.stop();
		std::cout << "**** FFT time: " << mytimer.elapsed_msec() << " ms." << std::endl;
		mytimer.start();
		compute_mod_mat(f_mat_i_);
		mytimer.stop();
		std::cout << "**** mod F time: " << mytimer.elapsed_msec() << " ms." << std::endl;
		mytimer.start();
		mask_mat(mask, 1 - mod_f_mat_i_);
		copy_mod_mat(1 - mod_f_mat_i_);
		mytimer.stop();
		std::cout << "**** Mask time: " << mytimer.elapsed_msec() << " ms." << std::endl;
		mytimer.start();
		compute_model_norm(1 - mod_f_mat_i_);
		mytimer.stop();
		std::cout << "**** model norm time: " << mytimer.elapsed_msec() << " ms." << std::endl;
		c_factor_ = base_norm / model_norm_;
		mytimer.start();
		prev_chi2_ = compute_chi2(pattern, 1 - mod_f_mat_i_, c_factor_);
		mytimer.stop();
		std::cout << "**** chi2 time: " << mytimer.elapsed_msec() << " ms." << std::endl;
		return true;
	} // Tile::init()


	bool Tile::copy_mod_mat(unsigned int src_i) {
#ifdef USE_GPU
		gtile_.copy_mod_mat(src_i);
#else
		mod_f_mat_[1 - src_i].populate(mod_f_mat_[src_i].data());
#endif // USE_GPU
		return true;
	} // Tile::copy_mod_mat()


	bool Tile::simulate_step(woo::Matrix2D<real_t>& pattern,
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
		update_fft_mat(dft_mat_, f_mat_[f_mat_i_], f_mat_[f_scratch_i], f_mat_i_, f_scratch_i);
		//print_cmatrix("f_mat_[f_scratch_i]", f_mat_[f_scratch_i].data(), size_, size_);
		compute_mod_mat(f_scratch_i);
		mask_mat(mask, mod_f_scratch_i);
		// this should be here i think ...
		compute_model_norm(mod_f_scratch_i);
		double new_c_factor = base_norm / model_norm_;
		double new_chi2 = compute_chi2(pattern, mod_f_scratch_i, new_c_factor);
		double diff_chi2 = prev_chi2_ - new_chi2;
		//std::cout << "++++ chi2 diff:\t" << prev_chi2_ << "\t" << new_chi2 << "\t" << diff_chi2
		//			<< "\t c_factor: " << new_c_factor << std::endl;

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
			compute_model_norm(mod_f_mat_i_);
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


	bool Tile::update_model(const woo::Matrix2D<real_t>& pattern, real_t base_norm) {
		update_a_mat();
#ifdef USE_GPU
		// not really needed!
		//cudaMemcpy(f_mat_[f_mat_i_], gtile_.f_mat_[f_mat_i_], size_ * size_, cudaMemcpyDeviceToHost);
		//cudaMemcpy(mod_f_mat_[mod_f_mat_i_], gtile_.mod_f_mat_[mod_f_mat_i_], size_ * size_,
		//			cudaMemcpyDeviceToHost);
#else
#endif // USE_GPU
		return true;
	} // Tile::update_model()


#ifndef USE_GPU // use CPU
	bool Tile::compute_fft_mat() {
		std::cout << "++ compute_fft_mat" << std::endl;

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
		#pragma omp parallel
		{
#ifdef _OPENMP
		if(omp_get_thread_num() == 0)
			std::cout << "[" << omp_get_num_threads() << " threads] ... " << std::flush;
#endif
		#pragma omp for collapse(2)
		for(unsigned int i = 0; i < size_; ++ i) {
			for(unsigned int j = 0; j < size_; ++ j) {
				f_mat_[f_mat_i_](i, j) = complex_t(mat_out[size_ * i + j][0], mat_out[size_ * i + j][1]) /
											(real_t)num_particles_;
			} // for
		} // for
		} // omp parallel

		//print_fftwcmatrix("mat_out", mat_out, size_, size_);
		//print_cmatrix("f_mat_[f_mat_i_]", f_mat_[f_mat_i_].data(), size_, size_);

		fftw_free(mat_out);
		fftw_free(mat_in);
		return true;
	} // Tile::compute_fft_mat_cpu()


	bool Tile::execute_fftw(fftw_complex* input, fftw_complex* output) {
		// create fft plan
		fftw_plan plan;
		plan = fftw_plan_dft_2d(size_, size_, input, output, FFTW_FORWARD, FFTW_ESTIMATE);
		fftw_execute(plan);
		// destroy fft plan
		fftw_destroy_plan(plan);
		return true;
	} // Tile::execute_cufft()

#else // USE_GPU

	bool Tile::compute_fft_mat() {
		std::cout << "++ compute_fft_mat_cuda" << std::endl;

		gtile_.compute_fft_mat(1 - f_mat_i_);
		gtile_.normalize_fft_mat(1 - f_mat_i_, num_particles_);
		return true;
	} // Tile::compute_fft_mat_cuda()
#endif // USE_GPU


	bool Tile::compute_mod_mat(unsigned int f_i) {
#ifdef USE_GPU
		return gtile_.compute_mod_mat(f_i, 1 - mod_f_mat_i_);
#else // USE CPU
		#pragma omp parallel for collapse(2)
		for(unsigned int i = 0; i < size_; ++ i) {
			for(unsigned int j = 0; j < size_; ++ j) {
				complex_t temp_f = f_mat_[f_i](i, j);
				mod_f_mat_[1 - mod_f_mat_i_](i, j) = temp_f.real() * temp_f.real() +
													temp_f.imag() * temp_f.imag();
			} // for
		} // for
		//print_matrix("mod_f_mat_[1 - mod_f_mat_i_]", mod_f_mat_[1 - mod_f_mat_i_].data(), size_, size_);
		return true;
#endif // USE_GPU
	} // Tile::compute_mod_mat()


	bool Tile::compute_model_norm(unsigned int buff_i) {
		double model_norm = 0.0;
		unsigned int maxi = size_ >> 1;
#ifdef USE_GPU
		model_norm = gtile_.compute_model_norm(buff_i);
#else
		#pragma omp parallel shared(model_norm)
		{
			#pragma omp for collapse(2) reduction(+:model_norm)
			for(unsigned int i = 0; i < maxi; ++ i) {
				for(unsigned int j = 0; j < maxi; ++ j) {
					model_norm += mod_f_mat_[buff_i](i, j) * (j + 1);	// what is the Y matrix anyway ???
				} // for
			} // for
		}
#endif // USE_GPU
		model_norm_ = model_norm;
		//std::cout << "++++ model_norm: " << model_norm_ << std::endl;
		return true;
	} // Tile::compute_model_norm()


	double Tile::compute_chi2(const woo::Matrix2D<real_t>& pattern, unsigned int mod_f_i, real_t c_factor) {
		double chi2 = 0.0;
#ifdef USE_GPU
		chi2 = gtile_.compute_chi2(mod_f_i, c_factor);
#else
		#pragma omp parallel for collapse(2) reduction(+:chi2)
		for(unsigned int i = 0; i < size_; ++ i) {
			for(unsigned int j = 0; j < size_; ++ j) {
				real_t temp = pattern(i, j) - mod_f_mat_[mod_f_i](i, j) * c_factor;
				chi2 += temp * temp;
			} // for
		} // for
#endif // USE_GPU
		return chi2;
	} // Tile::compute_chi2()


	bool Tile::virtual_move_random_particle() {
		old_pos_ = floor(ms_rand_01() * num_particles_);
		new_pos_ = floor(ms_rand_01() *	(size_ * size_ - num_particles_)) + num_particles_;
		old_index_ = indices_[old_pos_];
		new_index_ = indices_[new_pos_];
		//std::cout << "++++ old_pos,new_pos: " << old_pos_ << "," << new_pos_
		//			<< ", old_index,new_index: " << old_index_ << "," << new_index_ << std::endl;
		return true;
	} // Tile::virtual_move_random_particle()


	bool Tile::move_particle(double new_chi2, real_t base_norm) {
		// to swap buffers
		f_mat_i_ = 1 - f_mat_i_;
		mod_f_mat_i_ = 1 - mod_f_mat_i_;
		prev_chi2_ = new_chi2;

		// to swap indices (moving the particle)
		indices_[old_pos_] = new_index_;
		indices_[new_pos_] = old_index_;

		return true;
	} // Tile::move_particle()


	bool Tile::compute_dft2(woo::Matrix2D<complex_t> &vandermonde_mat, woo::Matrix2D<complex_t> &dft_mat) {
		//std::cout << "++ compute_dft2" << std::endl;
		unsigned int old_row = old_index_ / size_;
		unsigned int old_col = old_index_ % size_;
		unsigned int new_row = new_index_ / size_;
		unsigned int new_col = new_index_ % size_;
#ifdef USE_GPU
		gtile_.compute_dft2(old_row, old_col, new_row, new_col, num_particles_);
#else
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
#endif // USE_GPU
		return true;
	} // Tile::compute_dft2()


	bool Tile::update_fft_mat(woo::Matrix2D<complex_t> &dft_mat, woo::Matrix2D<complex_t> &f_mat,
								woo::Matrix2D<complex_t> &new_f_mat,
								unsigned int in_buff_i, unsigned int out_buff_i) {
#ifdef USE_GPU
		return gtile_.update_fft_mat(in_buff_i, out_buff_i);
#else
		return woo::matrix_add(dft_mat, f_mat, new_f_mat);	// matrix operation
#endif // USE_GPU
	} // Tile::update_fft_mat()


	bool Tile::mask_mat(const unsigned int*& mask_mat, unsigned int buff_i) {
#ifdef USE_GPU
		gtile_.mask_mat(buff_i);
#else
		#pragma omp parallel for collapse(2) shared(mask_mat)
		for(unsigned int i = 0; i < size_; ++ i) {
			for(unsigned int j = 0; j < size_; ++ j) {
				mod_f_mat_[buff_i](i, j) *= mask_mat[size_ * i + j];
			} // for
		} // for
#endif // USE_GPU
		return true;
	} // Tile::mask_mat()


	bool Tile::finalize_result(double& chi2, woo::Matrix2D<real_t>& a) {
		// populate a_mat_
		update_a_mat();
		a = a_mat_;
		chi2 = prev_chi2_;
		return true;
	} // Tile::finalize()


	bool Tile::update_a_mat() {
		a_mat_.fill(0.0);
		#pragma omp parallel for
		for(unsigned int i = 0; i < num_particles_; ++ i) {
			unsigned int x = indices_[i] / size_;
			unsigned int y = indices_[i] % size_;
			a_mat_(x, y) = 1.0;
		} // for
		return true;
	} // Tile::update_a_mat()

} // namespace hir
