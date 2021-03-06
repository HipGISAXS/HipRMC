/***
  *  Project: HipRMC - High Performance Reverse Monte Carlo Simulations
  *
  *  File: tile.hpp
  *  Created: Jan 25, 2013
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef __TILE_HPP__
#define __TILE_HPP__

#include <vector>
//#include <random>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <limits>

#ifndef USE_GPU
#include <fftw3.h>
#endif

#include "woo/timer/woo_boostchronotimers.hpp"
#include "woo/random/woo_mtrandom.hpp"

#include "wil/image.hpp"
#include "woo/comm/multi_node_comm.hpp"

#include "constants.hpp"
#include "typedefs.hpp"
#ifdef USE_GPU
#include "tile.cuh"
#endif

#include "hiprmc_input.hpp"
#include "tile_autotuner.hpp"

namespace hir {

	class Tile {

		private:
			// following define a tile
      unsigned int index_;          // index of the tile (useful when multiple tiles)
	  unsigned int seed_;			// random number generator seed for this tile
			unsigned int size_;						// current global tile size
			unsigned int rows_;						// number of local rows
			unsigned int cols_;						// number of local cols
			unsigned int final_size_;				// target model size (when using scaling)
			mat_real_t a_mat_;						// the current model

			unsigned int min_row_index_;			// global index of first local row
			unsigned int max_row_index_;			// global index of last local row
													// == min_row_index_ + a_mat_.num_rows() - 1
			#ifdef USE_MPI
				int row_offsets_[MAX_NUM_PROCS];
			#endif

			// buffers used only for cpu version
			std::vector<mat_complex_t> f_mat_;		// F buffers
			std::vector<mat_real_t> mod_f_mat_;		// auto_F buffers

			// used in both cpu and gpu versions
			std::vector<unsigned int> indices_;				// NOTE: the first num_particles_ entries
															// in indices_ are 'filled', rest are 'empty'
			unsigned int f_mat_i_;							// current f buffer index
			unsigned int mod_f_mat_i_;						// current mod_f buffer index
			real_t loading_factor_;							// loading factor
			real_t tstar_;									// temperature factor
			real_t cooling_factor_;							// cooling with iteration number
			bool tstar_set_;								// whether tstar was set previously
			unsigned int num_particles_;					// number of particles (duh!)
			unsigned int max_move_distance_;				// limit on particle movement

			#ifndef USE_GPU
				TileAutotuner autotuner_;					// autotuner
			#endif

			#ifdef USE_GPU
				cucomplex_t* cucomplex_buff_;
				GTile gtile_;
			#else
				fftw_complex *fft_in_;
				fftw_complex *fft_out_;
				fftw_plan fft_plan_;
			#endif // USE_GPU

			// following are used during simulation
			mat_complex_t dft_mat_;
			double prev_chi2_;

			std::vector<double> chi2_list_;						// stores all chi2, for plotting purposes
			unsigned int accepted_moves_;						// just for statistics

			// indices produced on virtually moving a particle
			unsigned int old_pos_;
			unsigned int new_pos_;
			unsigned int old_index_;
			unsigned int new_index_;

			// for random number generation
			woo::MTRandomNumberGenerator mt_rand_gen_;
			std::string prefix_;								// used as prefix to image filenames

			woo::BoostChronoTimer mytimer_;
			woo::BoostChronoTimer mytimer2_;

			// some temporary variables ...
			#ifndef USE_DFT
				mat_real_t virtual_a_mat_;			// the current virtual model
			#endif
			mat_real_t diff_mat_;					// mod difference matrix (temporary)

			// some timing variables
			real_t fft_update_time_;			// for dft or fft computation and update
			real_t reduction_time_;				// norm and chi2 computations
			real_t misc_time_;					// rest of the time
			real_t mpi_time_;					// all communications time
												// (note: this time overlaps with above times)


			// functions
			#ifdef USE_MPI
				bool compute_fft_mat(woo::MultiNode&);
				bool compute_fft_mat(unsigned int, woo::MultiNode&);
				bool compute_mod_mat(unsigned int, woo::MultiNode&);
				bool virtual_move_random_particle_restricted(unsigned int, woo::MultiNode&);
				bool move_particle(real_t, woo::MultiNode&);
			#else
				bool compute_fft_mat();
				bool compute_fft_mat(unsigned int);
				bool compute_mod_mat(unsigned int);
				bool virtual_move_random_particle_restricted(unsigned int);
				bool move_particle(real_t);
			#endif // USE_MPI
			#ifndef USE_GPU // use cpu
				bool execute_fftw(fftw_complex*, fftw_complex*);
			#endif
			//bool normalize_mod_mat(unsigned int);
			//bool compute_model_norm(unsigned int, const mat_uint_t&);
			//double compute_chi2(const mat_real_t&, unsigned int, real_t);
			//real_t compute_chi2(const mat_real_t&, unsigned int, const mat_uint_t&, real_t);
			bool virtual_move_random_particle();
			#ifdef USE_DFT
				//bool compute_dft2(mat_complex_t&, unsigned int, unsigned int);
				bool update_fft_mat(const mat_complex_t&, const mat_complex_t&, mat_complex_t&);
			#endif
			//bool mask_mat(const mat_uint_t&, unsigned int);
			bool copy_mod_mat(unsigned int);
			bool update_indices();
      bool accept_reject(real_t, real_t, real_t, real_t, unsigned int, unsigned int);

			// for autotuner (tstar)
			bool autotune_temperature(const mat_real_t&, mat_complex_t&, const mat_uint_t&, real_t, int
									#ifdef USE_MPI
										, woo::MultiNode&
									#endif
									);
			bool init_autotune(const mat_real_t&, const mat_uint_t&, real_t, real_t
									#ifdef USE_MPI
										, woo::MultiNode&
									#endif
									);
			bool simulate_autotune_step(const mat_real_t&, mat_complex_t&, const mat_uint_t&,
									real_t, unsigned int, unsigned int, unsigned int&
									#ifdef USE_MPI
										, woo::MultiNode&
									#endif
									);
			bool autotune_move_random_particle_restricted(unsigned int, unsigned int&, unsigned int&,
									unsigned int&, unsigned int&, unsigned int&, unsigned int&,
									unsigned int&, unsigned int&);
      bool save_acceptance_map(const std::map<const real_t, real_t>&i, const char*);
			#ifndef USE_GPU
				bool compute_fft(const mat_real_t&, mat_complex_t&);
				bool compute_mod(const mat_complex_t&, mat_real_t&);
				bool normalize(mat_real_t&);
			#endif // USE_GPU
			#ifdef USE_DFT
				bool compute_dft2(mat_complex_t&, unsigned int, unsigned int,
									unsigned int, unsigned int, mat_complex_t&
									#ifdef USE_MPI
										, woo::MultiNode&
									#endif
									);
				//bool update_fft(mat_complex_t&, mat_complex_t&);
			#endif
			real_t compute_chi2(const mat_real_t&, const mat_real_t&, const mat_uint_t&, real_t
									#ifdef USE_MPI
										, woo::MultiNode&
									#endif
									);

			// to record timings
			double vmove_time_, dft2_time_, mod_time_, norm_time_, chi2_time_;
			double rest_time_;

			void create_image(std::string, unsigned int, const mat_real_t&, bool);
			bool save_chi2_list();

		public:
			Tile(unsigned int, unsigned int, const std::vector<unsigned int>&, unsigned int, unsigned int,
				 unsigned int
           #ifdef USE_MPI
            , unsigned int
           #endif
           );
			Tile(const Tile&);
			~Tile();

			// initialize with raw data
			bool init(real_t, unsigned int, char*, int
									#ifdef USE_MPI
										, woo::MultiNode&
									#endif
									);
			bool init_scale(real_t, mat_real_t&, mat_complex_t&, mat_uint_t&, int
									#ifdef USE_MPI
										, woo::MultiNode&
									#endif
									);
			bool destroy_scale();
			// simulation functions
			bool simulate_step(const mat_real_t&, mat_complex_t&, const mat_uint_t&, real_t, unsigned int, unsigned int
									#ifdef USE_MPI
										, woo::MultiNode&
									#endif
									);
			bool update_model(
									#ifdef USE_MPI
										woo::MultiNode&
									#endif
							);
			#ifndef USE_DFT
				bool update_virtual_model();
			#endif
			bool update_a_mat();
			#ifdef USE_GPU
				bool update_f_mats();
			#endif
			bool update_from_model();
			bool print_times();
			bool print_new_times();
			bool finalize_result(double&
									#ifdef USE_MPI
										, woo::MultiNode&
									#endif
									);

			bool scale_step();
			bool print_a_mat();
			unsigned int accepted_moves() const { return accepted_moves_; }

			bool normalize_mod(unsigned int);

			bool clear_chi2_list();
			bool save_model() {
				create_image("fft", 0, mod_f_mat_[mod_f_mat_i_], true);
				create_image("final_model", 0, a_mat_, false);
				save_chi2_list();
				return true;
			} // save_model()

			// accessors
			real_t loading() const { return loading_factor_; }
			unsigned int size() const { return size_; }
      real_t temperature() const { if(tstar_set_) return tstar_; else return -1.0; }
      real_t cooling() const { return cooling_factor_; }

      // setters
      void temperature(real_t t) { tstar_ = t; tstar_set_ = true; }
      void cooling(real_t c) { cooling_factor_ = c; }

			// return a random number in (0,1)
			//real_t mt_rand_01() {
				//return ((real_t) (ms_rand_gen_() - ms_rand_gen_.min()) /
				//					(ms_rand_gen_.max() - ms_rand_gen_.min()));
			//	return mt_rand_gen_.rand();
			//} // ms_rand_01()


			double dft2_time() const { return dft2_time_; }

	}; // class Tile


	typedef std::vector<Tile> vec_tile_t;


} // namespace hir

#endif // __TILE_HPP__
