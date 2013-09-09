/***
  *  Project:
  *
  *  File: hiprmc_input.hpp
  *  Created: Jun 11, 2013
  *  Modified: Mon 09 Sep 2013 11:08:09 AM PDT
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef __HIPRMC_INPUT_HPP__
#define __HIPRMC_INPUT_HPP__

#include <stack>
#include <vector>
#include <unordered_map>
#include <string>

#include "tokens.hpp"
#include "token_mapper.hpp"
#include "read_oo_input.hpp"

namespace hir {

	// later on create a syntax tree out of the input reading
	// for that create a class with generic 'object type' and parent, children pointers
	// ...

	class HipRMCInput {

		private:

			/* containers */

			std::string input_image_;		// input image filename
			std::string mask_image_;		// mask image filename
			bool mask_set_;					// if the mask image filename is provided
			vec2_int_t image_size_;			// target image size
			unsigned int num_tiles_;		// this can be removed ...
			vec_real_t loading_factors_;	// loading factor of each tile
			vec_real_t tstar_;				// temperature for each tile
			vec_real_t cooling_factors_;	// cooling factor with iteration number for each tile

			vec2_int_t model_start_size_;	// size of starting model (when using scaling)
			unsigned int num_steps_factor_;	// this * image_size = total number of simulation steps
			unsigned int scale_factor_;		// number of rows and columns to increase at each scale step
			unsigned int max_move_distance_;	// distance limit on moving a particle, in pixels

			std::string label_;				// run name given by user

			/* helpers */

			Token curr_token_;
			Token past_token_;
			TokenType curr_keyword_;
			TokenType past_keyword_;

			std::stack <TokenType> keyword_stack_;	// for keyword tokens
													// keyword tokens get pushed on '{' and '['
													// and popped on '}' and ']'
			std::vector <real_t> curr_vector_;		// to store values in a vector while parsing it

			/**
			 * methods
			 */

			/* singleton */

			HipRMCInput();
			HipRMCInput(const HipRMCInput&);
			HipRMCInput& operator=(const HipRMCInput&);

			void init();

			/* setters */

			TokenType get_curr_parent();
			TokenType get_curr_grandparent();

			bool process_curr_keyword();
			bool process_curr_token();

			bool process_number(const real_t&);
			bool process_string(const std::string&);

			/* misc */

			inline bool preceeded_by_keyword() {
				return TokenMapper::instance().keyword_token_exists(past_token_.type_);
			} // HiGInput::preceeded_by_keyword()


			/* testers */
			// ...

		public:

			static HipRMCInput& instance() {
				static HipRMCInput hiprmc_input;
				return hiprmc_input;
			} // instance()

			bool construct_input_config(char* filename);

			/* getters */
			unsigned int num_rows() const { return image_size_[0]; }
			unsigned int num_cols() const { return image_size_[1]; }
			unsigned int num_tiles() const { return num_tiles_; }
			unsigned int model_start_num_rows() const { return model_start_size_[0]; }
			unsigned int model_start_num_cols() const { return model_start_size_[1]; }
			const char* input_image() const { return input_image_.c_str(); }
			const char* mask_image() const { return mask_image_.c_str(); }
			bool mask_set() const { return mask_set_; }
			vec_real_t loading() const { return loading_factors_; }
			vec_real_t tstar() const { return tstar_; }
			vec_real_t cooling() const { return cooling_factors_; }
			unsigned int num_steps_factor() const { return num_steps_factor_; }
			unsigned int scale_factor() const { return scale_factor_; }
			unsigned int max_move_distance() const { return max_move_distance_; }
			std::string label() const { return label_; }

			/* printing for testing */
			void print_all();

	}; // class HipRMCInput

} // namespace hir

#endif /* __HIPRMC_INPUT_HPP__ */
