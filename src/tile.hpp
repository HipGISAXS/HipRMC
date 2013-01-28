/***
  *  Project: HipRMC - High Performance Reverse Monte Carlo Simulations
  *
  *  File: tile.hpp
  *  Created: Jan 25, 2013
  *  Modified: Sun 27 Jan 2013 07:01:54 PM PST
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
			woo::Matrix2D<complex_t> a_mat_;
			woo::Matrix2D<complex_t> f_mat_;
			woo::Matrix2D<complex_t> mod_mat_;
			real_t loading_factor_;		// loading factor
			unsigned int num_particles_;
			std::vector<unsigned int> indices_;

			bool compute_fft_cuda();

		public:
			Tile(real_t, const std::vector<unsigned int>&);
			~Tile();

			// initialize with raw data
			bool init(real_t);

	}; // class Tile


	template <typename real_t, typename complex_t>
	Tile<real_t, complex_t>::Tile(unsigned int rows, unsigned int cols,
									const std::vector<unsigned int>& indices) :
		a_mat_(rows, cols),
		f_mat_(rows, cols),
		mod_mat_(rows, cols),
		indices_(indices) {

	} // Tile::Tile()


	template <typename real_t, typename complex_t>
	Tile<real_t, complex_t>::~Tile() {
	} // Tile::~Tile()


	// initialize with raw data
	template <typename real_t, typename complex_t>
	bool Tile<real_t, complex_t>::init(real_t loading) {
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

		// compute fft of a_mat_ into fft_mat_
		compute_fft_mat_cuda();
		compute_mod_mat();

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


} // namespace hir

#endif // __TILE_HPP__
