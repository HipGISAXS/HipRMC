/**
 *  Project: HipRMC
 *
 *  File: tile_autotuner.cpp
 *  Created: Sep 05, 2013
 *
 *  Author: Abhinav Sarje <asarje@lbl.gov>
 */

#include <map>

#include "tile.hpp"
#include "tile_autotuner.hpp"
#include "lw_solver.hpp"

namespace hir {

	TileAutotuner::TileAutotuner(unsigned int rows, unsigned int cols, const vec_uint_t& indices) :
		a_mat_(std::max(rows, cols), std::max(rows, cols)),
		size_(std::max(rows, cols)),
		indices_(indices),
		num_particles_(0),
		tstar_(0.0),
		cooling_factor_(0.0),
   		f_mat_(std::max(rows, cols), std::max(rows, cols)),
		mod_f_mat_(std::max(rows, cols), std::max(rows, cols)),
		dft_mat_(std::max(rows, cols), std::max(rows, cols)),
		prev_chi2_(0.0),
		accepted_moves_(0) {

	} // TileAutotuner::TileAutotuner()


	TileAutotuner::TileAutotuner(const TileAutotuner& a) :
		a_mat_(a.a_mat_.num_rows(), a.a_mat_.num_cols()),
		size_(std::max(a.a_mat_.num_rows(), a.a_mat_.num_cols())),
		num_particles_(a.num_particles_),
		tstar_(a.tstar_),
		cooling_factor_(a.cooling_factor_),
		f_mat_(a.a_mat_.num_rows(), a.a_mat_.num_cols()),
		mod_f_mat_(a.a_mat_.num_rows(), a.a_mat_.num_cols()),
		dft_mat_(a.a_mat_.num_rows(), a.a_mat_.num_cols()),
		prev_chi2_(a.prev_chi2_),
		accepted_moves_(a.accepted_moves_),
		indices_(a.indices_) {

		// TODO: copy a_mat_ ...

	} // TileAutotuner::TileAutotuner()


	TileAutotuner::~TileAutotuner() {

	} // TileAutotuner::~TileAutotuner()


	bool TileAutotuner::init(const vec_uint_t &indices, unsigned int num_particles, real_t tstar
							#ifdef USE_MPI
								, woo::MultiNode& multi_node
							#endif
							) {
		tstar_ = tstar;
		cooling_factor_ = 0.0;
		indices_ = indices;
		num_particles_ = num_particles;
		update_a_mat();				// update/construct a mat using indices
		accepted_moves_ = 0;
	} // TileAutotuner::init()


	bool TileAutotuner::update_a_mat() {
		a_mat_.fill(0.0);
		#pragma omp parallel for
		for(unsigned int i = 0; i < num_particles_; ++ i) {
			unsigned int x = indices_[i] / size_;
			unsigned int y = indices_[i] % size_;
			a_mat_(x, y) = 1.0;
		} // for

		return true;
	} // TileAutotuner::update_a_mat()


	/**
	 * autotuner functions from class Tile
	 */

	// autotune the temperature (tstar)
	bool Tile::autotune_temperature(const mat_real_t& pattern, mat_complex_t& vandermonde,
									const mat_uint_t& mask,
									real_t base_norm, int num_steps
									#ifdef USE_MPI
										, woo::MultiNode& multi_node
									#endif
									) {
		std::cout << "++ Autotuning temperature ..." << std::endl;
		std::map <const real_t, real_t> acceptance_map;
		real_t tmin = 0.0, tstep = 0.05, tmax = 2.0, tstar = tmin;
		unsigned int tstar_tune_steps = 50;		// input parameter TODO ...
		if(tstar_set_) tmax = tstar_;
		for(int i = 0; tstar < tmax + tstep; ++ i, tstar += tstep) {
			// simulate few steps with current tstar and obtain the acceptance rate
			if(!init_autotune(pattern, mask, tstar, base_norm
						#ifdef USE_MPI
							, multi_node
						#endif
						)) {
				std::cout << "error: failed to initialize autotuner" << std::endl;
				return false;
			} // if
			for(int iter = 0; iter < tstar_tune_steps; ++ iter) {
				simulate_autotune_step(pattern, vandermonde, mask, base_norm, iter
						#ifdef USE_MPI
							, multi_node
						#endif
						);
			} // for
			acceptance_map[tstar] = ((real_t) autotuner_.accepted_moves_) / tstar_tune_steps;
			//std::cout << "@@\t" << tstar << "\t" << acceptance_map[tstar] << std::endl;
		} // for

		// find min and max acceptance rate values
		real_t min_acc = 1.0, max_acc = 0.0;
		for(map_real_t::const_iterator i = acceptance_map.begin(); i != acceptance_map.end(); ++ i) {
			min_acc = (min_acc > (*i).second) ? (*i).second : min_acc;
			max_acc = (max_acc < (*i).second) ? (*i).second : max_acc;
		} // for
		// scale them
	//	for(map_real_t::iterator i = acceptance_map.begin(); i != acceptance_map.end(); ++ i) {
	//		(*i).second = ((*i).second - min_acc) / (max_acc - min_acc);
	//	} // for

		//std::cout << "@@@@@@@@@@@ MIN_ACC: " << min_acc << ", MAX_ACC: " << max_acc << std::endl;

		LWSolver lw;
		real_t a = 0.25, b = 0.25;	// initial parameter choices
		lw.solve_sigmoid(acceptance_map, a, b);
		//std::cout << "+++++++++++++ a: " << a << ", b: " << b << std::endl;

		//tstar_ = a + b / 2.0;
		//cooling_factor_ = (tstar_ / std::max(1e-5, a - b) - 1.0) / num_steps;

		//real_t temp_tstar = -0.8472979 * b + a;		// this is for acceptance = 0.3
		real_t temp_tstar = -2.1972246 * b + a;		// this is for acceptance = 0.1
													// derived from y = 1 / (1 + e^(- (x - a) / b))
													// implies x = b * (ln y - ln(1 - y)) + a
													// for y = 0.05, x =
													// for y = 0.1, x = -2.1972246 * b + a
													// for y = 0.2, x =
													// for y = 0.3, x = -0.8472979 * b + a
													// for y = 0.4, x =
													// for y = 0.5, x =
		temp_tstar = std::max(1e-2, std::min(temp_tstar, 1.0));
		if(tstar_set_) {
			// make sure it is lower or equal to previous tstar_:
			temp_tstar = std::min(tstar_, temp_tstar);
		} // if
		real_t cooling = std::max(1e-2, (temp_tstar / std::max(1e-5, a - b)) / num_steps);

		//std::cout << "@@\ta = " << a << ", b = " << b << std::endl;
		//std::cout << "@@\tTSTAR: " << tstar_ << ", COOLING: " << cooling_factor_
		//			<< ", TMIN: " << std::max(1e-2, a - b) << std::endl;
		//std::cout << "@@\tNEWTSTAR: " << temp_tstar << ", NEWCOOLING: " << cooling << std::endl;

		tstar_ = temp_tstar; cooling_factor_ = cooling; tstar_set_ = true;

		std::cout << "**                      Temperature: " << tstar_ << std::endl;
		std::cout << "**                          Cooling: " << cooling_factor_ << std::endl;

		return true;
	} // Tile::autotune_temperature()


	// to be executed at beginning of each autotune
	bool Tile::init_autotune(const mat_real_t& pattern, const mat_uint_t& mask,
								real_t tstar, real_t base_norm
								#ifdef USE_MPI
									, woo::MultiNode& multi_node
								#endif
								) {
		// initialize the autotuner
		autotuner_.init(indices_, num_particles_, tstar
						#ifdef USE_MPI
							, multi_node
						#endif
						);
		compute_fft(autotuner_.a_mat_, autotuner_.f_mat_);
		compute_mod(autotuner_.f_mat_, autotuner_.mod_f_mat_);
		// compute the initial chi2
		autotuner_.prev_chi2_ = compute_chi2(pattern, autotuner_.mod_f_mat_, mask, base_norm
						#ifdef USE_MPI
							, multi_node
						#endif
						);

		return true;
	} // Tile::init_autotune()


	bool Tile::simulate_autotune_step(const mat_real_t& pattern,
										mat_complex_t& vandermonde,
										const mat_uint_t& mask,
										real_t base_norm,
										unsigned int iter
										#ifdef USE_MPI
											, woo::MultiNode& multi_node
										#endif
										) {
		unsigned int old_x = 0, old_y = 0, new_x = 0, new_y = 0;
		unsigned int old_pos = 0, new_pos = 0, old_index = 0, new_index = 0;
		autotune_move_random_particle_restricted(max_move_distance_,
													old_pos, new_pos,
													old_index, new_index,
													old_x, new_x, old_y, new_y);
		compute_dft2(vandermonde, old_x, old_y, new_x, new_y, autotuner_.dft_mat_
					#ifdef USE_MPI
						, multi_node
					#endif
					);
		//update_fft(autotuner_.f_mat_, autotuner_.dft_mat_);
		update_fft_mat(autotuner_.dft_mat_, autotuner_.f_mat_, autotuner_.f_mat_);
		compute_mod(autotuner_.f_mat_, autotuner_.mod_f_mat_);
		double new_chi2 = compute_chi2(pattern, autotuner_.mod_f_mat_, mask, base_norm
					#ifdef USE_MPI
						, multi_node
					#endif
					);
		double diff_chi2 = autotuner_.prev_chi2_ - new_chi2;
		bool accept = false;
		if(diff_chi2 > 0.0) accept = true;
		else {
			real_t p = exp(diff_chi2 * (autotuner_.cooling_factor_ * iter + 1) / autotuner_.tstar_);
			real_t prand = mt_rand_gen_.rand();
			if(prand < p) accept = true;
		} // if-else
		if(accept) {	// accept the move
			++ autotuner_.accepted_moves_;
			autotuner_.prev_chi2_ = new_chi2;
			autotuner_.indices_[old_pos] = new_index;
			autotuner_.indices_[new_pos] = old_index;
		} // if

		return true;
	} // Tile::simulate_step()


	bool Tile::autotune_move_random_particle_restricted(unsigned int max_dist,
			unsigned int &old_pos, unsigned int &new_pos,
			unsigned int &old_index, unsigned int &new_index,
			unsigned int &old_x, unsigned int &new_x,
			unsigned int &old_y, unsigned int &new_y) {
		while(1) {
			old_pos = floor(mt_rand_gen_.rand() * num_particles_);
			new_pos = floor(mt_rand_gen_.rand() * (size_ * size_ - num_particles_)) + num_particles_;
			old_index = autotuner_.indices_[old_pos];
			new_index = autotuner_.indices_[new_pos];
			old_x = old_index / size_; old_y = old_index % size_;
			new_x = new_index / size_; new_y = new_index % size_;
			if((fabs(new_x - old_x) < max_dist || (size_ - fabs(new_x - old_x)) < max_dist) &&
					(fabs(new_y - old_y) < max_dist || (size_ - fabs(new_y - old_y)) < max_dist))
				return true;
		} // while
		return false;
	} // Tile::virtual_move_random_particle()


} // namespace hir
