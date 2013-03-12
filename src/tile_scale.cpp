/***
  *  Project:
  *
  *  File: tile_scale.cpp
  *  Created: Mar 04, 2013
  *  Modified: Sun 10 Mar 2013 02:12:35 PM PDT
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifdef _OPENMP
#include <omp.h>
#endif
#include <woo/timer/woo_boostchronotimers.hpp>

#include "tile.hpp"
#include "constants.hpp"
#include "temp.hpp"

namespace hir {

	bool Tile::scale_step() {
		//std::cout << "++ scale_step" << std::endl;
		std::cout << "---- initial loading: " << loading_factor_;

		// first update the model to make sure the latest model is in 'a'
		unsigned int new_row_par = 0, new_col_par = 0;
		std::vector<unsigned int> zero_indices, fill_indices;

		// get random indices for row and col where to insert new stuff
		unsigned int curr_size = size_;
		unsigned int new_row_i = floor(ms_rand_01() * curr_size);
		unsigned int new_col_i = floor(ms_rand_01() * curr_size);

		// construct new row and col
		real_t *new_row, *new_col;
		new_row = new (std::nothrow) real_t[curr_size];		// inserting row first
		new_col = new (std::nothrow) real_t[curr_size + 1];	// then insert col, hence size + 1

		// fill the new row
		// making it wrap around periodic
		mat_real_t::row_iterator prev_row, next_row;
		if(new_row_i != 0) prev_row = a_mat_.row(new_row_i - 1);
		else prev_row = a_mat_.row(curr_size - 1);
		if(new_row_i != curr_size) next_row = a_mat_.row(new_row_i);
		else next_row = a_mat_.row(0);

		for(unsigned int i = 0; i < curr_size; ++ i) {
			unsigned int prev_i = (i == 0) ? curr_size - 1 : i - 1;
			unsigned int next_i = (i == curr_size - 1) ? 0 : i + 1;
			real_t temp = floor((prev_row[prev_i] + 2 * prev_row[i] + prev_row[next_i] +
								next_row[prev_i] + 2 * next_row[i] + next_row[next_i]) / 8 + 0.5);
			if(temp < 0.5) {
				new_row[i] = 0.0;
				zero_indices.push_back(i);
			} else {
				new_row[i] = 1.0;
				++ new_row_par;
				fill_indices.push_back(i);
			} // if-else
		} // for
		unsigned int curr_num_par = floor(loading_factor_ * curr_size * curr_size) + new_row_par;
		unsigned int req_num_par = floor(loading_factor_ * curr_size * (curr_size + 1));
		if(req_num_par > curr_num_par) {
			// insert the remaining particles at random zero positions
			unsigned int insert_num_par = req_num_par - curr_num_par;
			if(insert_num_par > zero_indices.size()) { std::cerr << "STOP! STOP! STOP!" << std::endl; }
			std::random_device rd;
			std::mt19937_64 gen(rd());
			std::shuffle(zero_indices.begin(), zero_indices.end(), gen);
		//	std::cout << "INSERT ROW NUM PAR: " << insert_num_par << ", size: " << zero_indices.size()
		//				<< std::endl;
			for(unsigned int i = 0; i < insert_num_par; ++ i) new_row[zero_indices[i]] = 1.0;
			new_row_par += insert_num_par;
		} else if(req_num_par < curr_num_par) {
			// remove the remaining particles from random fill positions
			unsigned int remove_num_par = curr_num_par - req_num_par;
		//	std::cout << "+++++++++++++ REMOVE FROM ROW " << remove_num_par << std::endl;
			if(remove_num_par > fill_indices.size()) { std::cerr << "STOP! STOP! STOP!" << std::endl; }
			std::random_device rd;
			std::mt19937_64 gen(rd());
			std::shuffle(fill_indices.begin(), fill_indices.end(), gen);
		//	std::cout << "REMOVE ROW NUM PAR: " << remove_num_par << ", size: " << fill_indices.size()
		//				<< std::endl;
			for(unsigned int i = 0; i < remove_num_par; ++ i) new_row[fill_indices[i]] = 0.0;
			new_row_par -= remove_num_par;
		} // if-else

		//std::cout << "INSERTING ROW AT " << new_row_i << ": " << std::endl;
		//for(unsigned int i = 0; i < curr_size; ++ i) std::cout << new_row[i] << " ";
		//std::cout << std::endl;

		// insert the new row
		a_mat_.insert_row(new_row_i, new_row, curr_size);

		// now for the column:
		++ curr_size;
		zero_indices.clear();
		fill_indices.clear();
		
		// fill the new col
		// making it wrap around periodic
		mat_real_t::col_iterator prev_col, next_col;
		if(new_col_i != 0) prev_col = a_mat_.column(new_col_i - 1);
		else prev_col = a_mat_.column(curr_size - 1);
		if(new_col_i != curr_size) next_col = a_mat_.column(new_col_i);
		else next_col = a_mat_.column(0);
		for(unsigned int i = 0; i < curr_size; ++ i) {
			unsigned int prev_i = (i == 0) ? curr_size - 1 : i - 1;
			unsigned int next_i = (i == curr_size - 1) ? 0 : i + 1;
			real_t temp = floor((prev_col[prev_i] + 2 * prev_col[i] + prev_col[next_i] +
								next_col[prev_i] + 2 * next_col[i] + next_col[next_i]) / 8 + 0.5);
			if(temp < 0.5) {
				new_col[i] = 0.0;
				zero_indices.push_back(i);
			} else {
				new_col[i] = 1.0;
				++ new_col_par;
				fill_indices.push_back(i);
			} // if-else
		} // for
		curr_num_par = floor(loading_factor_ * curr_size * (curr_size - 1)) + new_col_par;
		req_num_par = floor(loading_factor_ * curr_size * curr_size);
		if(req_num_par > curr_num_par) {
			unsigned int insert_num_par = req_num_par - curr_num_par;
			if(insert_num_par > zero_indices.size()) { std::cerr << "STOP! STOP! STOP!" << std::endl; }
			std::random_device rd2;
			std::mt19937_64 gen2(rd2());
			std::shuffle(zero_indices.begin(), zero_indices.end(), gen2);
		//	std::cout << "INSERT COL NUM PAR: " << insert_num_par << ", size: " << zero_indices.size()
		//				<< std::endl;
			for(unsigned int i = 0; i < insert_num_par; ++ i) new_col[zero_indices[i]] = 1.0;
			new_col_par += insert_num_par;
		} else if(req_num_par < curr_num_par) {
			// remove the remaining particles from random fill positions
			unsigned int remove_num_par = curr_num_par - req_num_par;
		//	std::cout << "+++++++++++++ REMOVE FROM COL " << remove_num_par << std::endl;
			if(remove_num_par > fill_indices.size()) { std::cerr << "STOP! STOP! STOP!" << std::endl; }
			std::random_device rd;
			std::mt19937_64 gen(rd());
			std::shuffle(fill_indices.begin(), fill_indices.end(), gen);
		//	std::cout << "REMOVE COL NUM PAR: " << remove_num_par << ", size: " << fill_indices.size()
		//				<< std::endl;
			for(unsigned int i = 0; i < remove_num_par; ++ i) new_col[fill_indices[i]] = 0.0;
			new_col_par -= remove_num_par;
		} // if-else

		//std::cout << "INSERTING COL AT " << new_col_i << ": " << std::endl;
		//for(unsigned int i = 0; i < curr_size; ++ i) std::cout << new_col[i] << " ";
		//std::cout << std::endl;

		// insert the new col
		a_mat_.insert_col(new_col_i, new_col, curr_size);

		// this is just temporary for testing ...
		curr_num_par = floor(loading_factor_ * (curr_size - 1) * (curr_size - 1)) + new_row_par + new_col_par;
		real_t new_loading = curr_num_par / (real_t)(curr_size * curr_size);
		//std::cout << "\tnew_par: " << new_row_par + new_col_par;
		std::cout << "\tfinal loading: " << new_loading << std::endl;
		loading_factor_ = new_loading;

		// update indices array and num_particles
		update_indices();
		// update sizes of the f and mod f matrices, dft mat
		f_mat_[0].incr_rows(1); f_mat_[0].incr_columns(1);
		f_mat_[1].incr_rows(1); f_mat_[1].incr_columns(1);
		mod_f_mat_[0].incr_rows(1); mod_f_mat_[0].incr_columns(1);
		mod_f_mat_[1].incr_rows(1); mod_f_mat_[1].incr_columns(1);
		dft_mat_.incr_rows(1); dft_mat_.incr_columns(1);
		// clear chi2 list
		chi2_list_.clear();

		// update size
		++ size_;

		delete[] new_col;
		delete[] new_row;

		return true;
	} // Tile::scale_step()

} // namespace hir
