/***
  *  Project: HipRMC - High Performance Reverse Monte Carlo Simulations
  *
  *  File: tile.hpp
  *  Created: Jan 25, 2013
  *  Modified: Sun 03 Feb 2013 08:39:32 PM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef __TILE_HPP__
#define __TILE_HPP__

#include <vector>
#include <random>
#include <woo/matrix/matrix.hpp>
#ifndef USE_GPU
#include <fftw3.h>
#endif // USE_GPU

#include "typedefs.hpp"
#ifdef USE_GPU
#include "tile.cuh"
#endif // USE_GPU

namespace hir {

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

#ifdef USE_GPU
			// buffers for device
			cucomplex_t* a_mat_d_;								// not really needed
			cucomplex_t* f_mat_d_[2];
			real_t* mod_f_mat_d_[2];
			cucomplex_t* complex_buffer_h_;
			real_t* real_buffer_h_;
			// cuda parameters
			unsigned int block_x_;
			unsigned int block_y_;
			unsigned int grid_x_;
			unsigned int grid_y_;
			// new stuff:
			GTile gtile_;
#endif // USE_GPU

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
			bool compute_fft_mat();
#ifndef USE_GPU // use cpu (possibly openmp)
			bool execute_fftw(fftw_complex*, fftw_complex*);			// cpu
#else // use cuda
			bool execute_cufft(cuFloatComplex*, cuFloatComplex*);		// gpu
			bool execute_cufft(cuDoubleComplex*, cuDoubleComplex*);		// gpu
#endif
			bool compute_mod_mat(const woo::Matrix2D<complex_t>&);
			bool compute_model_norm(const woo::Matrix2D<real_t>&);
			double compute_chi2(const woo::Matrix2D<real_t>&, const woo::Matrix2D<real_t>&, real_t);
			bool virtual_move_random_particle();
			bool move_particle(double, real_t);
			bool compute_dft2(woo::Matrix2D<complex_t>&, woo::Matrix2D<complex_t>&);
			bool update_fft_mat(woo::Matrix2D<complex_t>&, woo::Matrix2D<complex_t>&,
								woo::Matrix2D<complex_t>&);
			bool mask_mat(const unsigned int*&, const woo::Matrix2D<real_t>&);
#ifdef USE_GPU
			bool normalize_fft_mat(cucomplex_t*, unsigned int);
#endif // USE_GPU

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


} // namespace hir

#endif // __TILE_HPP__
