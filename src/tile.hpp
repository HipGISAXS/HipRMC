/***
  *  Project: HipRMC - High Performance Reverse Monte Carlo Simulations
  *
  *  File: tile.hpp
  *  Created: Jan 25, 2013
  *  Modified: Mon 11 Mar 2013 11:38:11 AM PDT
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef __TILE_HPP__
#define __TILE_HPP__

#include <vector>
#include <random>
#include <sstream>
#include <iomanip>
#include <fstream>

#ifndef USE_GPU
#include <fftw3.h>
#endif

#include <woo/timer/woo_boostchronotimers.hpp>

//#include <woo/visual/image.hpp>
#include "wil/image.hpp"

#include "typedefs.hpp"
#ifdef USE_GPU
#include "tile.cuh"
#endif

namespace hir {

	class Tile {

		private:
			// following define a tile
			unsigned int size_;						// num rows = num cols = size
			unsigned int final_size_;				// target model size (when using scaling)
			mat_real_t a_mat_;						// A: the current model

			// buffers used only for cpu version
			std::vector<mat_complex_t> f_mat_;		// F buffers
			std::vector<mat_real_t> mod_f_mat_;		// auto_F buffers

			// used in both cpu and gpu versions
			std::vector<unsigned int> indices_;					// NOTE: the first num_particles_ entries
																// in indices_ are 'filled', rest are 'empty'
			unsigned int f_mat_i_;								// current f buffer index
			unsigned int mod_f_mat_i_;							// current mod_f buffer index
			real_t loading_factor_;								// loading factor
			unsigned int num_particles_;						// number of particles (duh!)
			double model_norm_;									// norm of current model
			double c_factor_;									// c factor

			#ifdef USE_GPU
				cucomplex_t* cucomplex_buff_;
				GTile gtile_;
			#endif // USE_GPU

			// following are used during simulation
			mat_complex_t dft_mat_;
			double prev_chi2_;

			std::vector<double> chi2_list_;						// stores all chi2, for plotting purposes

			// indices produced on virtually moving a particle
			unsigned int old_pos_;
			unsigned int new_pos_;
			unsigned int old_index_;
			unsigned int new_index_;

			// for random number generation
			std::mt19937_64 ms_rand_gen_;

			woo::BoostChronoTimer mytimer_;
			double vmove_time, dft2_time, mod_time, norm_time, chi2_time;
			double rest_time;


			// functions
			bool compute_fft_mat();
			#ifndef USE_GPU // use cpu
				bool execute_fftw(fftw_complex*, fftw_complex*);
			#endif
			bool compute_mod_mat(unsigned int);
			bool compute_model_norm(unsigned int);
			//double compute_chi2(const mat_real_t&, unsigned int, real_t);
			double compute_chi2(const mat_real_t&, unsigned int, real_t);
			bool virtual_move_random_particle();
			bool move_particle(double, real_t);
			bool compute_dft2(mat_complex_t&, unsigned int, unsigned int);
			bool update_fft_mat(mat_complex_t&, mat_complex_t&,
								mat_complex_t&, unsigned int, unsigned int);
			bool mask_mat(const mat_uint_t&, unsigned int);
			bool copy_mod_mat(unsigned int);
			bool update_indices();

		public:
			Tile(unsigned int, unsigned int, const std::vector<unsigned int>&, unsigned int);
			Tile(const Tile&);
			~Tile();

			// initialize with raw data
			bool init(real_t, real_t, mat_real_t&, const mat_complex_t&, mat_uint_t&);
			bool init_scale(real_t, mat_real_t&, const mat_complex_t&, mat_uint_t&);
			// simulation functions
			bool simulate_step(mat_real_t&, mat_complex_t&, const mat_uint_t&, real_t, real_t);
			bool update_model();
			bool update_a_mat();
			#ifdef USE_GPU
				bool update_f_mats();
			#endif
			bool update_from_model();
			bool print_times();
			bool finalize_result(double&, mat_real_t&);

			bool scale_step();
			bool print_a_mat();

			bool save_mat_image(unsigned int i) {
				wil::Image img(mod_f_mat_[mod_f_mat_i_].num_rows(), mod_f_mat_[mod_f_mat_i_].num_cols());
				real_t* data = mod_f_mat_[mod_f_mat_i_].data();
				img.construct_image(data);
				std::string str("_modf.tif");
				std::stringstream num;
				num << std::setfill('0') << std::setw(4) << i;
				char str0[5];
				num >> str0;
				str = std::string(str0) + str;
				img.save(str);
			} // save_mat_image()

			bool save_fmat_image(unsigned int i) {
				unsigned int nrows = f_mat_[f_mat_i_].num_rows(), ncols = f_mat_[f_mat_i_].num_cols();
				wil::Image img(nrows, ncols);
				real_t* data = new (std::nothrow) real_t[nrows * ncols];
				for(unsigned int i = 0; i < nrows; ++ i) {
					for(unsigned int j = 0; j < ncols; ++ j) {
						real_t temp = f_mat_[f_mat_i_](i, j).real();
						data[i * ncols + j] = temp * temp;
					} // for
				} // for
				img.construct_image(data);
				std::string str("_f.tif");
				std::stringstream num;
				num << std::setfill('0') << std::setw(4) << i;
				char str0[5];
				num >> str0;
				str = std::string(str0) + str;
				img.save(str);
				delete[] data;
			} // save_mat_image()

			bool save_mat_image_direct(unsigned int i) {
				wil::Image img(a_mat_.num_rows(), a_mat_.num_cols());
				img.construct_image_direct(a_mat_.data());
				std::string str("_a_mat.tif");
				std::stringstream num;
				num << std::setfill('0') << std::setw(4) << i;
				char str0[5];
				num >> str0;
				str = std::string(str0) + str;
				img.save(str);
			} // save_mat_image_direct()

			bool save_chi2_list() {
				std::ofstream chi2out("chi2_list.dat", std::ios::out | std::ios::app);
				for(unsigned int i = 0; i < chi2_list_.size(); ++ i) {
					chi2out << i << "\t" << chi2_list_[i] << std::endl;
				} // for
				chi2out.close();
			} // save_chi2_list()

			// accessors
			real_t loading() const { return loading_factor_; }
			unsigned int size() const { return size_; }

			// return a random number in (0,1)
			real_t ms_rand_01() {
				return ((real_t) (ms_rand_gen_() - ms_rand_gen_.min()) /
									(ms_rand_gen_.max() - ms_rand_gen_.min()));
			} // ms_rand_01()

	}; // class Tile


	typedef std::vector<Tile> vec_tile_t;


} // namespace hir

#endif // __TILE_HPP__
