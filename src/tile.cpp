/***
  *  Project: HipRMC - High Performance Reverse Monte Carlo Simulations
  *
  *  File: tile.cpp
  *  Created: Jan 25, 2013
  *  Modified: Wed 13 Mar 2013 01:05:26 PM PDT
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifdef USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif // USE_GPU
#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP
#include <opencv2/opencv.hpp>
#include <woo/timer/woo_boostchronotimers.hpp>

#include "tile.hpp"
#include "constants.hpp"
#include "temp.hpp"

namespace hir {

	// constructor
	Tile::Tile(unsigned int rows, unsigned int cols, const std::vector<unsigned int>& indices,
				unsigned int final_size) :
		a_mat_(rows, cols),
		f_mat_i_(0),
		mod_f_mat_i_(0),
		indices_(indices),
		dft_mat_(rows, cols) {

		woo::BoostChronoTimer mytimer;

		size_ = std::max(rows, cols);
		final_size_ = final_size;

		// two buffers each
		mytimer.start();
		f_mat_.push_back(mat_complex_t(size_, size_));
		f_mat_.push_back(mat_complex_t(size_, size_));
		mytimer.stop();
		std::cout << "***** f_mat_: " << mytimer.elapsed_msec() << " ms." << std::endl;
		mytimer.start();
		mod_f_mat_.push_back(mat_real_t(size_, size_));
		mod_f_mat_.push_back(mat_real_t(size_, size_));
		mytimer.stop();
		std::cout << "***** mod_f_mat_: " << mytimer.elapsed_msec() << " ms." << std::endl;
		#ifdef USE_GPU
			// device memory allocation takes all the time
			mytimer.start();
			unsigned int size2 = final_size_ * final_size_;
			cucomplex_buff_ = new (std::nothrow) cucomplex_t[size2];
			mytimer.stop();
			std::cout << "***** device mem: " << mytimer.elapsed_msec() << " ms." << std::endl;
		#endif // USE_GPU
	} // Tile::Tile()


	// copy constructor
	Tile::Tile(const Tile& tile):
		size_(tile.size_),
		final_size_(tile.final_size_),
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
			unsigned int size2 = final_size_ * final_size_;
			cucomplex_buff_ = new (std::nothrow) cucomplex_t[size2];
		#endif // USE_GPU
	} // Tile::Tile()


	Tile::~Tile() {
		#ifdef USE_GPU
			delete[] cucomplex_buff_;
		#endif // USE_GPU
	} // Tile::~Tile()


	// initialize with raw data
	bool Tile::init(real_t loading, real_t base_norm, mat_real_t& pattern,
					const mat_complex_t& vandermonde, mat_uint_t& mask) {
		woo::BoostChronoTimer mytimer;
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
			gtile_.init(pattern.data(), cucomplex_buff_, a_mat_.data(), mask.data(), final_size_, size_,
						block_x, block_y);
			//print_cucmatrix("vandermonde", cucomplex_buff_, size_, size_);
		#endif // USE_GPU

		// compute fft of a_mat_ into fft_mat_ and other stuff
/*		mytimer.start();
		compute_fft_mat();
		mytimer.stop();
		std::cout << "**** FFT time: " << mytimer.elapsed_msec() << " ms." << std::endl;
		mytimer.start();
		compute_mod_mat(f_mat_i_);
		mytimer.stop();
		std::cout << "**** mod F time: " << mytimer.elapsed_msec() << " ms." << std::endl;
		mytimer.start();
		#ifndef USE_GPU
			mask_mat(mask, 1 - mod_f_mat_i_);
		#endif // USE_GPU
		copy_mod_mat(1 - mod_f_mat_i_);
		mytimer.stop();
		std::cout << "**** Mask/copy time: " << mytimer.elapsed_msec() << " ms." << std::endl;
		mytimer.start();
		compute_model_norm(1 - mod_f_mat_i_);
		mytimer.stop();
		std::cout << "**** model norm time: " << mytimer.elapsed_msec() << " ms." << std::endl;
		c_factor_ = base_norm / model_norm_;
		mytimer.start();
		prev_chi2_ = compute_chi2(pattern, 1 - mod_f_mat_i_, c_factor_);
		std::cout << "++++ initial chi2 = " << prev_chi2_ << std::endl;
		mytimer.stop();
		std::cout << "**** chi2 time: " << mytimer.elapsed_msec() << " ms." << std::endl;*/
		return true;
	} // Tile::init()


	// to be executed at simulation beginning after a scaling
	bool Tile::init_scale(real_t base_norm, mat_real_t& pattern, const mat_complex_t& vandermonde,
							mat_uint_t& mask) {
		if(pattern.num_rows() != size_ || vandermonde.num_rows() != size_ || mask.num_rows() != size_) {
			std::cerr << "error: some matrix size is not what it should be! check your BUGGY code!"
						<< std::endl;
			return false;
		} // if
		// compute fft of a_mat_ into fft_mat_ and other stuff
		woo::BoostChronoTimer mytimer;
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
			gtile_.init_scale(pattern.data(), cucomplex_buff_, a_mat_.data(), mask.data(),
								size_, block_x, block_y);
			//print_cucmatrix("vandermonde", cucomplex_buff_, size_, size_);
		#endif // USE_GPU
		mytimer.start();
		compute_fft_mat();
		mytimer.stop();
		std::cout << "**** FFT time: " << mytimer.elapsed_msec() << " ms." << std::endl;
		mytimer.start();
		compute_mod_mat(f_mat_i_);
		mytimer.stop();
		std::cout << "**** mod F time: " << mytimer.elapsed_msec() << " ms." << std::endl;
		mytimer.start();
		#ifndef USE_GPU
			mask_mat(mask, 1 - mod_f_mat_i_);
		#endif // USE_GPU
		copy_mod_mat(1 - mod_f_mat_i_);
		mytimer.stop();
		std::cout << "**** Mask/copy time: " << mytimer.elapsed_msec() << " ms." << std::endl;
		mytimer.start();
		compute_model_norm(1 - mod_f_mat_i_);
		mytimer.stop();
		std::cout << "**** model norm time: " << mytimer.elapsed_msec() << " ms." << std::endl;
		c_factor_ = base_norm / model_norm_;
		mytimer.start();
		prev_chi2_ = compute_chi2(pattern, 1 - mod_f_mat_i_, c_factor_);
		std::cout << "++++ initial chi2 = " << prev_chi2_ << std::endl;
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


	// in case of gpu version, this assumes all data is already on the gpu
	bool Tile::simulate_step(mat_real_t& pattern,
							mat_complex_t& vandermonde,
							const mat_uint_t& mask,
							real_t tstar, real_t base_norm) {
		//std::cout << "++ simulate_step" << std::endl;
		// do all computations in scratch buffers
		unsigned int f_scratch_i = 1 - f_mat_i_;
		unsigned int mod_f_scratch_i = 1 - mod_f_mat_i_;

		mytimer_.start();
		virtual_move_random_particle();
		mytimer_.stop(); vmove_time += mytimer_.elapsed_msec();
		mytimer_.start();
		compute_dft2(vandermonde, f_mat_i_, f_scratch_i);
		#ifndef USE_GPU
			update_fft_mat(dft_mat_, f_mat_[f_mat_i_], f_mat_[f_scratch_i], f_mat_i_, f_scratch_i);
		#endif // USE_GPU
		mytimer_.stop(); dft2_time += mytimer_.elapsed_msec();
		//print_cmatrix("f_mat_[f_scratch_i]", f_mat_[f_scratch_i].data(), size_, size_);
		mytimer_.start();
		compute_mod_mat(f_scratch_i);
		#ifndef USE_GPU
			mask_mat(mask, mod_f_scratch_i);
		#endif // USE_GPU
		mytimer_.stop(); mod_time += mytimer_.elapsed_msec();
		// this should be here i think ...
		mytimer_.start();
		compute_model_norm(mod_f_scratch_i);
		mytimer_.stop(); norm_time += mytimer_.elapsed_msec();
		mytimer_.start();
		double new_c_factor = base_norm / model_norm_;
		double new_chi2 = compute_chi2(pattern, mod_f_scratch_i, new_c_factor);
		double diff_chi2 = prev_chi2_ - new_chi2;
		mytimer_.stop(); chi2_time += mytimer_.elapsed_msec();
		//std::cout << "++++ chi2 diff:\t" << prev_chi2_ << "\t" << new_chi2 << "\t" << diff_chi2
		//			<< "\t c_factor: " << new_c_factor << std::endl;

		mytimer_.start();
		bool accept = false;
		if(new_chi2 < prev_chi2_) accept = true;
		else {
			real_t p = exp(diff_chi2 / tstar);
			real_t prand = ms_rand_01();
			if(prand < p) accept = true;
		} // if-else
		if(accept) {	// accept the move
			//std::cout << std::endl << "++++ accepting move..., chi2: " << new_chi2 << ", prev: " << prev_chi2_
			//			<< ", old cf: " << c_factor_ << std::endl;
			// update to newly computed stuff
			// make scratch as current
			move_particle(new_chi2, base_norm);
			c_factor_ = new_c_factor;
			chi2_list_.push_back(new_chi2);		// save this chi2 value
		} // if
		mytimer_.stop(); rest_time += mytimer_.elapsed_msec();

		return true;
	} // Tile::simulate_step()


	bool Tile::update_model() {
		update_a_mat();
		return true;
	} // Tile::update_model()


	bool Tile::update_from_model() {
		update_indices();
		return true;
	} // Tile::update_from_model()


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
					std::cout << "[" << omp_get_num_threads() << " threads] ... "
								<< std::endl << std::flush;
			#endif
			#pragma omp for collapse(2)
			for(unsigned int i = 0; i < size_; ++ i) {
				for(unsigned int j = 0; j < size_; ++ j) {
					f_mat_[f_mat_i_](i, j) = complex_t(mat_out[size_ * i + j][0], mat_out[size_ * i + j][1]) /
												(real_t) num_particles_;
				} // for
			} // for
		} // omp parallel

		//print_fftwcmatrix("mat_out", mat_out, size_, size_);
		//print_cmatrix("f_mat_[f_mat_i_]", f_mat_[f_mat_i_].data(), size_, size_);

		fftw_free(mat_out);
		fftw_free(mat_in);
		return true;
	} // Tile::compute_fft_mat()


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

		gtile_.compute_fft_mat(f_mat_i_);
		gtile_.normalize_fft_mat(f_mat_i_, num_particles_);
		return true;
	} // Tile::compute_fft_mat()

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
		unsigned int maxi = size_;// >> 1;
		#ifdef USE_GPU
			model_norm = gtile_.compute_model_norm(buff_i);
		#else
			#pragma omp parallel shared(model_norm)
			{
				#pragma omp for collapse(2) reduction(+:model_norm)
				for(unsigned int i = 0; i < maxi; ++ i) {
					for(unsigned int j = 0; j < maxi; ++ j) {
						model_norm += mod_f_mat_[buff_i](i, j); // * (j + 1);	// what is the Y matrix anyway ???
					} // for
				} // for
			}
		#endif // USE_GPU
		model_norm_ = model_norm;
		//std::cout << "++++ model_norm: " << model_norm_ << std::endl;
		return true;
	} // Tile::compute_model_norm()


	double Tile::compute_chi2(const mat_real_t& pattern, unsigned int mod_f_i, real_t c_factor) {
		double chi2 = 0.0;
		#ifdef USE_GPU
			chi2 = gtile_.compute_chi2(mod_f_i, c_factor);
		#else
			#pragma omp parallel for collapse(2) reduction(+:chi2)
			for(unsigned int i = 0; i < size_; ++ i) {
				for(unsigned int j = 0; j < size_; ++ j) {
					real_t temp = pattern(i, j) - mod_f_mat_[mod_f_i](i, j) * c_factor;
					//std::cout << "--------- pattern: " << pattern(i, j) << ", mod_f: "
					//			<< mod_f_mat_[mod_f_i](i, j) << ", c_fac: " << c_factor
					//			<< ", chi: " << temp << std::endl;
					chi2 += temp * temp;
				} // for
			} // for
		#endif // USE_GPU
		// normalize with the size
		chi2 = chi2 / (size_ * size_);
		return chi2;
	} // Tile::compute_chi2()


	/*double Tile::compute_chi2(mat_real_t& pattern, unsigned int mod_f_i, real_t c_factor) {
		double chi2 = 0.0;
		//#ifdef USE_GPU
		//	chi2 = gtile_.compute_chi2(mod_f_i, c_factor);
		//#else
		//	#pragma omp parallel for collapse(2) reduction(+:chi2)
			unsigned int psize = pattern.num_rows();
			real_t* pdata = new (std::nothrow) real_t[psize * psize];
			memcpy(pdata, pattern.data(), psize * psize * sizeof(real_t));
			real_t* spdata;
			wil::scale_image((int)psize, (int)psize, (int)size_, (int)size_, pdata, spdata);
			for(unsigned int i = 0; i < size_; ++ i) {
				for(unsigned int j = 0; j < size_; ++ j) {
					real_t temp = spdata[size_ * i + j] - mod_f_mat_[mod_f_i](i, j) * c_factor;
					chi2 += temp * temp;
				} // for
			} // for
			//delete[] spdata;
		//#endif // USE_GPU
		return chi2;
	} // Tile::compute_chi2()*/


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


	bool Tile::compute_dft2(mat_complex_t& vandermonde_mat,
							unsigned int in_buff_i, unsigned int out_buff_i) {
		//std::cout << "++ compute_dft2" << std::endl;
		unsigned int old_row = old_index_ / size_;
		unsigned int old_col = old_index_ % size_;
		unsigned int new_row = new_index_ / size_;
		unsigned int new_col = new_index_ % size_;
		#ifdef USE_GPU
			gtile_.compute_dft2(old_row, old_col, new_row, new_col, num_particles_,
								in_buff_i, out_buff_i);
		#else
			typename mat_complex_t::row_iterator old_row_iter = vandermonde_mat.row(old_row);
			typename mat_complex_t::col_iterator old_col_iter = vandermonde_mat.column(old_col);
			typename mat_complex_t::row_iterator new_row_iter = vandermonde_mat.row(new_row);
			typename mat_complex_t::col_iterator new_col_iter = vandermonde_mat.column(new_col);
			#pragma omp parallel for collapse(2)
			for(unsigned int row = 0; row < size_; ++ row) {
				for(unsigned int col = 0; col < size_; ++ col) {
					complex_t new_temp = new_col_iter[row] * new_row_iter[col];
					complex_t old_temp = old_col_iter[row] * old_row_iter[col];
					dft_mat_(row, col) = (new_temp - old_temp) / (real_t)num_particles_;
				} // for
			} // for
			//print_cmatrix("dft_mat", dft_mat.data(), size_, size_);
		#endif // USE_GPU

		return true;
	} // Tile::compute_dft2()


	bool Tile::update_fft_mat(mat_complex_t& dft_mat, mat_complex_t& f_mat,
								mat_complex_t& new_f_mat,
								unsigned int in_buff_i, unsigned int out_buff_i) {
		#ifdef USE_GPU
			// this has been merged into compute_dft2 for gpu
		#else
			return woo::matrix_add(dft_mat, f_mat, new_f_mat);	// matrix operation
		#endif // USE_GPU
	} // Tile::update_fft_mat()


	bool Tile::mask_mat(const mat_uint_t& mask_mat, unsigned int buff_i) {
		#ifdef USE_GPU
			/// this has been merged into compute_mod_mat for gpu
		#else
			#pragma omp parallel for collapse(2) shared(mask_mat)
			for(unsigned int i = 0; i < size_; ++ i) {
				for(unsigned int j = 0; j < size_; ++ j) {
					mod_f_mat_[buff_i](i, j) *= mask_mat(i, j);
				} // for
			} // for
		#endif // USE_GPU
		return true;
	} // Tile::mask_mat()


	bool Tile::finalize_result(double& chi2, mat_real_t& a) {
		// populate a_mat_
		update_a_mat();
		#ifdef USE_GPU
			// also copy f mat data to main memory from GPU
			update_f_mats();
		#endif
		a = a_mat_;
		chi2 = prev_chi2_;
		return true;
	} // Tile::finalize()


	// update a_mat_ using indices_
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


#ifdef USE_GPU
	// update the f and modf matrices on the host with data from device
	bool Tile::update_f_mats() {
		unsigned int size2 = size_ * size_;
		real_t *mod_f_buff = new (std::nothrow) real_t[size2];
		complex_t *f_buff = new (std::nothrow) complex_t[size2];
		if(mod_f_buff == NULL || f_buff == NULL) {
			std::cout << "error: could not allocate memory for f buffers in update_f_mats" << std::endl;
			return false;
		} // if

		gtile_.copy_f_mats_to_host(cucomplex_buff_, mod_f_buff, f_mat_i_, mod_f_mat_i_);
		for(unsigned int i = 0; i < size2; ++ i)
			f_buff[i] = complex_t(cucomplex_buff_[i].x, cucomplex_buff_[i].y);
		f_mat_[f_mat_i_].populate(f_buff);
		mod_f_mat_[mod_f_mat_i_].populate(mod_f_buff);

		delete[] f_buff;
		delete[] mod_f_buff;
		return true;
	} // Tile::update_f_mats()
#endif


	// update indices, num_particles and loading_factor using a_mat_
	bool Tile::update_indices() {
		unsigned int rows = a_mat_.num_rows();
		unsigned int cols = a_mat_.num_cols();
		indices_.clear();
		std::vector<unsigned int> zero_indices;
		num_particles_ = 0;
		for(unsigned int i = 0; i < rows * cols; ++ i) {
			if(a_mat_[i] > 0.5) { indices_.push_back(i); ++ num_particles_; }
			else zero_indices.push_back(i);
		} // for
		indices_.insert(indices_.end(), zero_indices.begin(), zero_indices.end());
		//loading_factor_ = (real_t) num_particles_ / (rows * cols);
		std::cout << "+++++++++++++++ actual loading: " << (real_t) num_particles_ / (rows * cols)
					<< std::endl;
		return true;
	} // Tile::update_indices()


	bool Tile::print_times() {
		std::cout << "vmove time: " << vmove_time << std::endl;
		std::cout << "dft2 time: " << dft2_time << std::endl;
		std::cout << "mod time: " << mod_time << std::endl;
		std::cout << "norm time: " << norm_time << std::endl;
		std::cout << "chi2 time: " << chi2_time << std::endl;
		std::cout << "rest time: " << rest_time << std::endl;
		return true;
	} // Tile::print_times()


	bool Tile::print_a_mat() {
		print_matrix("a_mat", a_mat_.data(), a_mat_.num_rows(), a_mat_.num_cols());
	} // Tile::print_a_mat()


} // namespace hir
