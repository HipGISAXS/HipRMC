/***
  *  Project: HipRMC - High Performance Reverse Monte Carlo Simulations
  *
  *  File: tile.hpp
  *  Created: Jan 25, 2013
  *  Modified: Mon 28 Jan 2013 03:30:10 PM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef __TILE_HPP__
#define __TILE_HPP__

#include <opencv2/opencv.hpp>
#include <woo/matrix/matrix.hpp>

namespace hir {

	template <typename real_t, typename complex_t>
	class Tile {
		private:
			// following define a tile
			woo::Matrix2D<complex_t> a_mat_;		// A
			woo::Matrix2D<complex_t> f_mat_[2];		// F buffers
			unsigned int f_mat_i_;					// current buffer index
			woo::Matrix2D<real_t> mod_f_mat_[2];	// auto_F buffers
			unsigned int mod_f_mat_i_;				// current buffer index
			real_t loading_factor_;					// loading factor
			unsigned int num_particles_;
			std::vector<unsigned int> indices_;
			real_t model_norm_;
			real_t c_factor_;

			// following are used during simulation
			real_t prev_chi2_;
			real_t curr_chi2_;
			unsigned int prev_pos_;
			unsigned int curr_pos_;
			unsigned int prev_index_;
			unsigned int curr_index_;

			bool compute_fft_mat_cuda();
			bool compute_mod_mat();
			bool compute_model_norm();

		public:
			Tile(unsigned int, unsigned int, const std::vector<unsigned int>&, const woo::Matrix2D<real_t>&);
			~Tile();

			// initialize with raw data
			bool init(real_t, real_t, const woo::Matrix2D<real_t>&);
			bool compute_chi2(const woo::Matrix2D<real_t>&);

	}; // class Tile


	template <typename real_t, typename complex_t>
	Tile<real_t, complex_t>::Tile(unsigned int rows, unsigned int cols,
									const std::vector<unsigned int>& indices,
									const woo::Matrix2D<real_t>&) :
		a_mat_(rows, cols),
		f_mat_[0](rows, cols), f_mat_[1](rows, cols), f_mat_i_(0),
		mod_f_mat_[0](rows, cols), mod_f_mat_[1](rows, cols), mod_f_mat_i_(0),
		indices_(indices) {

	} // Tile::Tile()


	template <typename real_t, typename complex_t>
	Tile<real_t, complex_t>::~Tile() {
	} // Tile::~Tile()


	// initialize with raw data
	template <typename real_t, typename complex_t>
	bool Tile<real_t, complex_t>::init(real_t loading, real_t base_norm,
										const woo::Matrix2D<real_t>& pattern) {
		loading_factor_ = loading;
		unsigned int cols = a_mat_.num_cols(), rows = a_mat_.num_rows();
		num_particles_ = ceil(loading * rows * cols);
		// NOTE: the first num_particles_ entries in indices_ are filled, rest are empty

		// fill a_mat_ with particles
		for(unsigned int i = 0; i < num_particles_; ++ i) {
			unsigned int x = indices[i] / cols;
			unsigned int y = indices[i] % cols;
			a_mat_(x, y) = 1.0;
		} // for

		// compute fft of a_mat_ into fft_mat_ and other stuff
		compute_fft_mat_cuda();
		compute_mod_mat();
		compute_model_norm();
		c_factor_ = base_norm / model_norm_;
		prev_chi2_ = compute_chi2(pattern);

		return true;
	} // Tile::init()


	template<typename real_t, typename complex_t>
	bool Tile<real_t, complex_t>::compute_fft_mat_cuda() {
		// TODO ...
	} // Tile::compute_fft_mat_cuda()


	template<typename real_t, typename complex_t>
	bool Tile<real_t, complex_t>::compute_mod_mat() {
		// TODO ...
	} // Tile::compute_mod_mat_cuda()


	template<typename real_t, typename complex_t>
	bool Tile<real_t, complex_t>::compute_mod_mat_cuda() {
		// TODO ...
	} // Tile::compute_mod_mat_cuda()


	template<typename real_t, typename complex_t>
	bool Tile<real_t, complex_t>::compute_model_norm() {
		// TODO ...
	} // Tile::compute_model_norm()


	template<typename real_t, typename complex_t>
	bool Tile<real_t, complex_t>::compute_chi2(const woo::Matrix2D<real_t>&) {
		// TODO ...
	} // Tile::compute_chi2()


	template<typename real_t, typename complex_t>
	bool Tile<real_t, complex_t>::simulate_step(const woo::Matrix2D<real_t>& pattern,
												const woo::Matrix2D<real_t>& vandermonde,
												real_t tstar) {
		// do all computations in scratch buffers
		f_scratch_i = 1 - f_mat_i_;
		mod_f_scratch_i = 1 - mod_f_mat_i_;

		move_random_particle();
		compute_dft2();
		update_fft_mat();
		compute_mod_mat();
		mask_fft_mat();
		curr_chi2_ = compute_chi2(pattern);
		real_t diff_chi2 = prev_chi2_ - curr_chi2_;
		real_t p = std::min(1, exp(diff_chi2 / tstar));
		real_t prand = rand();

		if(prand <= p) {	// accept the move
			// update to newly computed stuff
			// make scratch as current
			f_mat_i_ = f_scratch_i;
			mod_f_mat_i_ = mod_f_scratch_i;
			prev_chi2_ = curr_chi2_;
		} // if

		return true;
	} // Tile::simulate_step()


	template<typename real_t, typename complex_t>
	bool Tile<real_t, complex_t>::update_model() {
		compute_model_norm();
		c_factor_ = base_norm / model_norm_;
		prev_chi2_ = compute_chi2(pattern);

		return true;
	} // Tile::update_model()


	template<typename real_t, typename complex_t>
	bool Tile<real_t, complex_t>::move_random_particle() {
		// TODO ...
	} // Tile::move_random_particle()


	template<typename real_t, typename complex_t>
	bool Tile<real_t, complex_t>::compute_dft2() {
		// TODO ...
	} // Tile::compute_dft2()

} // namespace hir

#endif // __TILE_HPP__
