/***
  *  Project: HipRMC - High Performance Reverse Monte Carlo Simulations
  *
  *  File: tile.hpp
  *  Created: Jan 25, 2013
  *  Modified: Tue 29 Jan 2013 02:01:09 PM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef __TILE_HPP__
#define __TILE_HPP__

#include <opencv2/opencv.hpp>
#include <woo/matrix/matrix.hpp>

namespace hir {

	template <typename real_t, typename complex_t, typename cucomplex_t>
	class Tile {
		private:
			// following define a tile
			unsigned int size_;						// num rows = num cols = size
			woo::Matrix2D<real_t> a_mat_;		// A
			woo::Matrix2D<complex_t> f_mat_[2];		// F buffers
			unsigned int f_mat_i_;					// current buffer index
			woo::Matrix2D<real_t> mod_f_mat_[2];	// auto_F buffers
			unsigned int mod_f_mat_i_;				// current buffer index
			real_t loading_factor_;					// loading factor
			unsigned int num_particles_;
			std::vector<unsigned int> indices_;		// NOTE: the first num_particles_ entries
													// in indices_ are 'filled', rest are 'empty'
			real_t model_norm_;
			real_t c_factor_;

			// buffers for device
			cucomplex_t* a_mat_d_;
			cucomplex_t* f_mat_d_;

			// following are used during simulation
			real_t prev_chi2_;
			real_t curr_chi2_;
			// indices produced on virtually moving a particle
			unsigned int old_pos_;
			unsigned int new_pos_;
			unsigned int old_index_;
			unsigned int new_index_;

			// functions
			bool compute_fft_mat_cuda();
			bool execute_cufft(cuFloatComplex_t*&, cuFloatComplex_t*&);
			bool execute_cufft(cuDoubleComplex_t*&, cuDoubleComplex_t*&);
			bool compute_mod_mat();
			bool compute_model_norm();
			bool compute_chi2(const woo::Matrix2D<real_t>&, const woo::Matrix2D<real_t>&, real_t);
			bool virtual_move_random_particle();
			bool move_particle();
			bool compute_dft2(const woo::Matrix2D<complex_t>&, woo::Matrix2D<complex_t>&);
			bool update_fft_mat(const woo::Matrix2D<complex_t>&, const woo::Matrix2D<complex_t>&,
								woo::Matrix2D<complex_t>&);
			bool mask_mat(const unsigned int*&, const woo::Matrix2D<real_t>&);

		public:
			Tile(unsigned int, unsigned int, const std::vector<unsigned int>&, const woo::Matrix2D<real_t>&);
			~Tile();

			// initialize with raw data
			bool init(real_t, real_t, const woo::Matrix2D<real_t>&);
			bool simulate_step(const woo::Matrix2D<real_t>&, const woo::Matrix2D<real_t>&,
								unsigned int*&, real_t);
			bool update_model(const woo::Matrix2D<real_t>& pattern, real_t base_norm);
			bool finalize();

	}; // class Tile


	template <typename real_t, typename complex_t, typename cucomplex_t>
	Tile<real_t, complex_t, cucomplex_t>::Tile(unsigned int rows, unsigned int cols,
									const std::vector<unsigned int>& indices) :
		a_mat_(rows, cols),
		f_mat_[0](rows, cols), f_mat_[1](rows, cols), f_mat_i_(0),
		mod_f_mat_[0](rows, cols), mod_f_mat_[1](rows, cols), mod_f_mat_i_(0),
		indices_(indices) {

		size_ = std::max(rows, cols);
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
													const woo::Matrix2D<real_t>& pattern) {
		srand(time(NULL));
		loading_factor_ = loading;
		unsigned int cols = a_mat_.num_cols(), rows = a_mat_.num_rows();
		num_particles_ = ceil(loading * rows * cols);
		// NOTE: the first num_particles_ entries in indices_ are filled, rest are empty

		// fill a_mat_ with particles
		a_mat_.fill(0.0);
		for(unsigned int i = 0; i < num_particles_; ++ i) {
			unsigned int x = indices_[i] / cols;
			unsigned int y = indices_[i] % cols;
			a_mat_(x, y) = 1.0;
		} // for

		// compute fft of a_mat_ into fft_mat_ and other stuff
		compute_fft_mat_cuda();
		compute_mod_mat();
		compute_model_norm();
		c_factor_ = base_norm / model_norm_;
		prev_chi2_ = compute_chi2(pattern, f_mod_mat_[1 - f_mod_mat_i_], c_factor_);
		return true;
	} // Tile::init()


	template<typename real_t, typename complex_t, typename cucomplex_t>
	bool Tile<real_t, complex_t, cucomplex_t>::compute_fft_mat_cuda() {
		unsigned int size2 = size_ * size_;
		cucomplex_t* temp_mat = new (std::nothrow) cucomplex_t[size2];
		real_t* orig_a_mat = a_mat_.data();
		for(int i = 0; i < size2; ++ i) {
			temp_a_mat[i].x = orig_a_mat[i];
			temp_a_mat[i].y = 0.0;
		} // for
		// copy data to device
		cudaMemcpy(a_mat_d_, temp_mat, size2 * sizeof(cucomplex_t), cudaMemcpyHostToDevice);
		// execute fft
		execute_cufft(a_mat_d_, f_mat_d_);
		// copy data to host, reuse the temp_mat buffer
		cudaMemcpy(temp_mat, f_mat_d_, size2 * sizeof(cucomplex_t), cudaMemcpyDeviceToHost);
		f_mat_[f_mat_i_].populate(temp_mat);

		delete[] temp_mat;
		return true;
	} // Tile::compute_fft_mat_cuda()


	template<typename real_t, typename complex_t, typename cucomplex_t>
	bool Tile<real_t, complex_t, cucomplex_t>::execute_cufft(cuFloatComplex_t* &input,
															cuFloatComplex_t* &output) {
		// create fft plan
		cufftHandle plan;
		cufftPlan2d(&plan, size_, size_, CUFFT_C2C);
		cufftExecC2C(plan, input, output, CUFFT_FORWARD);
		cudaThreadSynchronize();
		// destroy fft plan
		cufftDestroy(plan);
		return true;
	} // Tile::execute_cufft()


	template<typename real_t, typename complex_t, typename cucomplex_t>
	bool Tile<real_t, complex_t, cucomplex_t>::execute_cufft(cuDoubleComplex_t* &input,
															cuDoubleComplex_t* &output) {
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
	bool Tile<real_t, complex_t, cucomplex_t>::compute_mod_mat() {
		for(unsigned int i = 0; i < size_; ++ i) {
			for(unsigned int j = 0; j < size_; ++ j) {
				complex_t temp_f = f_mat_[f_mat_i_](i, j);
				mod_f_mat_[1 - mod_f_mat_i_](i, j) = temp_f.real() * temp_f.real() +
													temp_f.imag() * temp_f.imag();
			} // for
		} // for
		return true;
	} // Tile::compute_mod_mat()


	template<typename real_t, typename complex_t, typename cucomplex_t>
	bool Tile<real_t, complex_t, cucomplex_t>::compute_model_norm() {
		model_norm_ = 0.0;
		for(unsigned int i = 0; i < size_; ++ i) {
			for(unsigned int j = 0; j < size_; ++ j) {
				model_norm_ += mod_f_mat_[mod_f_mat_i_](i, j) * (j + 1);
			} // for
		} // for
		return true;
	} // Tile::compute_model_norm()


	template<typename real_t, typename complex_t, typename cucomplex_t>
	real_t Tile<real_t, complex_t, cucomplex_t>::compute_chi2(const woo::Matrix2D<real_t>& pattern,
															const woo::Matrix2D<real_t>& mod_f,
															real_t c_factor) {
		real_t chi2 = 0.0;
		for(unsigned int i = 0; i < size_; ++ i) {
			for(unsigned int j = 0; j < size_; ++ j) {
				real_t temp = pattern(i, j) - mod_f(i, j) * c_factor;
				chi2 += temp * temp;
			} // for
		} // for
		return chi2;
	} // Tile::compute_chi2()


	template<typename real_t, typename complex_t, typename cucomplex_t>
	bool Tile<real_t, complex_t, cucomplex_t>::simulate_step(const woo::Matrix2D<real_t>& pattern,
												const woo::Matrix2D<real_t>& vandermonde,
												unsigned int* &mask,
												real_t tstar) {
		// do all computations in scratch buffers
		f_scratch_i = 1 - f_mat_i_;
		mod_f_scratch_i = 1 - mod_f_mat_i_;

		woo::Matrix2D<complex_t> dft_mat(size_, size_);

		virtual_move_random_particle();
		// merge the following two: ...
		compute_dft2(vandermonde, dft_mat);
		update_fft_mat(dft_mat, f_mat_[f_mat_i_], f_mat_[f_scratch_i_]);
		compute_mod_mat();
		mask_mat(mask, mod_f_mat_[mod_f_scratch_i_]);
		curr_chi2_ = compute_chi2(pattern, mod_f_mat_[mod_f_scratch_i_], c_factor_);
		real_t diff_chi2 = prev_chi2_ - curr_chi2_;
		real_t p = std::min(1, exp(diff_chi2 / tstar));
		real_t prand = (real_t)rand() / RAND_MAX;
		if(prand <= p) {	// accept the move
			// update to newly computed stuff
			// make scratch as current
			move_particle();
		} // if

		return true;
	} // Tile::simulate_step()


	template<typename real_t, typename complex_t, typename cucomplex_t>
	bool Tile<real_t, complex_t, cucomplex_t>::update_model(const woo::Matrix2D<real_t>& pattern,
															real_t base_norm) {
		compute_model_norm();
		c_factor_ = base_norm / model_norm_;
		prev_chi2_ = compute_chi2(pattern);
		return true;
	} // Tile::update_model()


	template<typename real_t, typename complex_t, typename cucomplex_t>
	bool Tile<real_t, complex_t, cucomplex_t>::virtual_move_random_particle() {
		old_pos_ = floor(((real_t)rand() / RAND_MAX) * num_particles_);
		new_pos_ = floor(((real_t)rand() / RAND_MAX) * (size_ * size_ - num_particles_)) +
									num_particles_;
		old_index_ = indices_[old_pos_];
		new_index_ = indices_[new_pos_];
		return true;
	} // Tile::virtual_move_random_particle()


	template<typename real_t, typename complex_t, typename cucomplex_t>
	bool Tile<real_t, complex_t, cucomplex_t>::move_particle() {
		// to swap buffers
		f_mat_i_ = 1 - f_mat_i;
		mod_f_mat_i_ = 1 - mod_f_mat_i;
		prev_chi2_ = curr_chi2_;

		// to swap indices
		indices_[old_pos_] = new_index_;
		indices_[new_pos_] = old_index_;

		return true;
	} // Tile::move_particle()


	template<typename real_t, typename complex_t, typename cucomplex_t>
	bool Tile<real_t, complex_t, cucomplex_t>::compute_dft2(const woo::Matrix2D<complex_t> &vandermonde_mat,
															woo::Matrix2D<complex_t> &dft_mat) {
		unsigned int old_row = old_index_ % size_;
		unsigned int old_col = old_index_ / size_;
		unsigned int new_row = new_index_ % size_;
		unsigned int new_col = new_index_ / size_;
		woo::Matrix2D<complex_t>::row_iterator old_row_iter = vandermonde_mat.row(old_row);
		woo::Matrix2D<complex_t>::col_iterator old_col_iter = vandermonde_mat.col(old_col);
		woo::Matrix2D<complex_t>::row_iterator new_row_iter = vandermonde_mat.row(new_row);
		woo::Matrix2D<complex_t>::col_iterator new_col_iter = vandermonde_mat.col(new_col);
		for(unsigned int row = 0; row < size_; ++ row) {
			for(unsigned int col = 0; col < size_; ++ col) {
				complex_t new_temp = new_col_iter[row] * new_row_iter[col];
				complex_t old_temp = old_col_iter[row] * old_row_iter[col];
				dft_mat(row, col) = (new_temp - old_temp) / num_particles_;
			} // for
		} // for
		return true;
	} // Tile::compute_dft2()


	template<typename real_t, typename complex_t, typename cucomplex_t>
	bool Tile<real_t, complex_t, cucomplex_t>::update_fft_mat(const woo::Matrix2D<complex_t> &dft_mat,
																const woo::Matrix2D<complex_t> &f_mat,
																woo::Matrix2D<complex_t> &new_f_mat) {
		return woo::matrix_add(dft_mat, f_mat, new_f_mat);	// matrix operation
	} // Tile::update_fft_mat()


	template<typename real_t, typename complex_t, typename cucomplex_t>
	bool Tile<real_t, complex_t, cucomplex_t>::mask_mat(const unsigned int*& mask_mat,
														const woo::Matrix2D<real_t>& mat) {
		for(unsigned int i = 0; i < size_; ++ i) {
			for(unsigned int j = 0; j < size_; ++ j) {
				mat(i, j) *= mask_mat[size_ * i + j];
			} // for
		} // for
		return true;
	} // Tile::mask_mat()


	template<typename real_t, typename complex_t, typename cucomplex_t>
	bool Tile<real_t, complex_t, cucomplex_t>::finalize(real_t& chi2, woo::Matrix2D<real_t>& a) {
		// populate a_mat_
		a_mat_.fill(0.0);
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
