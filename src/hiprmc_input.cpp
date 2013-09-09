/***
  *  Project:
  *
  *  File: hiprmc_input.cpp
  *  Created: Jun 11, 2013
  *  Modified: Mon 09 Sep 2013 10:52:21 AM PDT
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#include <cfloat>

#include "hiprmc_input.hpp"
#include "config_file_reader.hpp"
#include "utilities.hpp"


namespace hir {


	HipRMCInput::HipRMCInput() {
		/* instantiaize mapper, reader */
		InputReader::instance();
		TokenMapper::instance();
		ConfigFileReader::instance();
	} // HipRMCInput::HipRMCInput()


	void HipRMCInput::init() {
		// assign default values
		image_size_[0] = 0; image_size_[1] = 0;
		num_tiles_ = 1;
		loading_factors_.clear();
		tstar_.clear();
		cooling_factors_.clear();
		model_start_size_[0] = 0; model_start_size_[1] = 0;
		num_steps_factor_ = 100;
		scale_factor_ = 1;
		max_move_distance_ = 0;
	} // HipRMCInput::init();


	bool HipRMCInput::construct_input_config(char* filename) {
		if(!InputReader::instance().read_input(filename)) {
			std::cerr << "fatal error: some error happened in opening or reading "
						<< "input config file. aborting"
						<< std::endl;
			return false;
		} // if

		// parse all tokens (whole config file)
		curr_keyword_ = null_token; past_keyword_ = null_token;
		curr_token_ = InputReader::instance().get_next_token();
		past_token_.type_ = null_token;
		while(curr_token_.type_ != null_token) {
			if(curr_token_.type_ == error_token) {
				std::cerr << "aborting due to fatal error" << std::endl;
				return false;
			} // if
			if(!process_curr_token()) {
				std::cerr << "aborting due to fatal error" << std::endl;
				return false;
			} // if
			past_token_ = curr_token_;
			curr_token_ = InputReader::instance().get_next_token();
		} // while

		// set defaults in case they were not included in the config
		if(model_start_size_[0] == 0 || model_start_size_[1] == 0) model_start_size_ = image_size_;
		if(max_move_distance_ == 0) max_move_distance_ = image_size_[0];	// default: full freedom
		if(loading_factors_.size() != num_tiles_) {
			std::cerr << "error: check your number of tiles and all loading factors" << std::endl;
			return false;
		} // if
		if(tstar_.size() == 0) {
			for(int i = 0; i < num_tiles_; ++ i) tstar_.push_back(0.0);
		} else if(tstar_.size() == 1) {
			std::cout << "warning: a temperature value is given. setting it for all tiles. "
						<< "it may not be actually used due to temperature autotuning." << std::endl;
			for(int i = 0; i < num_tiles_ - 1; ++ i) tstar_.push_back(tstar_[0]);
		} else {
			std::cout << "warning: temperatures given. ignoring" << std::endl;
		} // if-else
		if(cooling_factors_.size() == 0) {
			for(int i = 0; i < num_tiles_; ++ i) cooling_factors_.push_back(0.0);
		} else if(cooling_factors_.size() == 1) {
			std::cout << "warning: a cooling factor value is given. setting it for all tiles. "
						<< "it may not be actually used due to temperature autotuning." << std::endl;
			for(int i = 0; i < num_tiles_ - 1; ++ i) cooling_factors_.push_back(cooling_factors_[0]);
		} else {
			std::cout << "warning: cooling factors given. ignoring" << std::endl;
		} // if

		return true;
	} // HipRMCInput::construct_input_config()


	bool HipRMCInput::process_curr_token() {
		TokenType parent = null_token;
		TokenType gparent = null_token;

		// process the token, do some basic syntax checking (improve with AST later) ...
		switch(curr_token_.type_) {

			case error_token:
				std::cerr << "aborting due to error" << std::endl;
				return false;

			case null_token:
				std::cerr << "error: something went wrong - should have already stopped!"
							<< std::endl;
				return false;

			case white_space_token:
				std::cerr << "error: something went wrong - "
							<< "seeing whitespace when not supposed to!" << std::endl;
				return false;

			case comment_token:	// do nothing
				return true;

			case object_begin_token:	// should be preceeded by '='
				if(past_token_.type_ != assignment_token && past_token_.type_ != comment_token) {
					std::cerr << "fatal error: unexpected object begin token '{'"
								<< std::endl;
					return false;
				} // if
				keyword_stack_.push(curr_keyword_);
				break;

			case object_end_token:		// preceeded by number or string or '}'
				if(past_token_.type_ != number_token &&
						past_token_.type_ != string_token &&
						past_token_.type_ != object_end_token &&
						past_token_.type_ != array_end_token &&
						past_token_.type_ != object_begin_token &&
						past_token_.type_ != comment_token) {
					std::cerr << "fatal error: unexpected object close token '}'" << std::endl;
					return false;
				} // if
				if(keyword_stack_.size() < 1) {
					std::cerr << "fatal error: unexpected object close token '}'. "
								<< "no matching object open token found" << std::endl;
					return false;
				} // if

				parent = get_curr_parent();
				switch(parent) {
					case hiprmc_token:
						// TODO ...
						break;

					case instrument_token:
						// TODO ...
						break;

					case compute_token:
						// TODO ...
						break;

					default:
						std::cerr << "error: something is wrong with one of your objects"
									<< std::endl;
						return false;
				} // switch
				if(keyword_stack_.size() < 1) {
					std::cerr << "something is really wrong. keyword_stack_ is empty when "
					   			<< "object end was found" << std::endl;
					return false;
				} // if
				past_keyword_ = curr_keyword_;
				curr_keyword_ = keyword_stack_.top();
				keyword_stack_.pop();
				break;

			case array_begin_token:	// should be preceeded by '='
				if(past_token_.type_ != assignment_token) {
					std::cerr << "fatal error: unexpected array begin token '['"
								<< std::endl;
					return false;
				} // if
				keyword_stack_.push(curr_keyword_);
				break;

			case array_end_token:	// preceeded by number_token or array_begin_token
				if(past_token_.type_ != number_token &&
						past_token_.type_ != array_begin_token &&
						past_token_.type_ != comment_token) {
					std::cerr << "fatal error: unexpected array close token ']'"
								<< std::endl;
					return false;
				} // if
				if(keyword_stack_.size() < 1) {
					std::cerr << "fatal error: unexpected array close token ']', "
								<< "no matching array open token found" << std::endl;
					return false;
				} // if

				parent = keyword_stack_.top();
				switch(parent) {
					case instrument_image_size_token:
						if(curr_vector_.size() != 2) {
							std::cerr << "error: there should be 2 numbers for imagesize" << std::endl;
							return false;
						} // if
						image_size_[0] = curr_vector_[0];
						image_size_[1] = curr_vector_[1];
						break;

					case instrument_loading_token:
						if(curr_vector_.size() < 1) {
							std::cerr << "error: there should be at least one tile" << std::endl;
							return false;
						} // if
						for(vec_real_t::iterator i = curr_vector_.begin(); i != curr_vector_.end(); ++ i) {
							if(*i > 1.0) {
								std::cerr << "error: loading factor cannot be greater than 1.0" << std::endl;
								return false;
							} // if
							loading_factors_.push_back(*i);
						} // for
						num_tiles_ = loading_factors_.size();
						break;

					case instrument_tstar_token:
						if(curr_vector_.size() < 1) {
							std::cerr << "error: there should be temperature for each tile" << std::endl;
							return false;
						} // if
						for(vec_real_t::iterator i = curr_vector_.begin(); i != curr_vector_.end(); ++ i) {
							if(*i < 0.0) {
								std::cerr << "error: temperature cannot be negative" << std::endl;
								return false;
							} // if
							tstar_.push_back(*i);
						} // for
						break;

					case instrument_cooling_factor_token:
						if(curr_vector_.size() < 1) {
							std::cerr << "error: there should be a cooling factor for each tile" << std::endl;
							return false;
						} // if
						for(vec_real_t::iterator i = curr_vector_.begin(); i != curr_vector_.end(); ++ i) {
							if(*i < 0.0) {
								std::cerr << "error: cooling factor cannot be negative" << std::endl;
								return false;
							} // if
							cooling_factors_.push_back(*i);
						} // for
						break;

					case compute_model_start_size_token:
						if(curr_vector_.size() != 2) {
							std::cerr << "error: there should be 2 numbers for modelstartsize" << std::endl;
							return false;
						} // if
						model_start_size_[0] = curr_vector_[0];
						model_start_size_[1] = curr_vector_[1];
						break;

					default:
						std::cerr << "error: found array value in place of non-array type" << std::endl;
						return false;
				} // switch
				curr_vector_.clear();
				keyword_stack_.pop();
				past_keyword_ = curr_keyword_;
				curr_keyword_ = keyword_stack_.top();
				break;

			case assignment_token:	// should be preceeded by a 'keyword',
									// followed by '{' (object) or '[' (array)
									// or string or number
				if(!preceeded_by_keyword()) {
					std::cerr << "error: misplaced assignment token '='" << std::endl;
					return false;
				} // if
				break;

			case number_token:		// preceeded by '=' or '[' or number_token
				if(past_token_.type_ != assignment_token &&
						past_token_.type_ != array_begin_token &&
						past_token_.type_ != number_token &&
						past_token_.type_ != comment_token &&
						past_token_.type_ != white_space_token) {
					std::cerr << "error: unexpected number '"
								<< curr_token_.dvalue_ << "'" << std::endl;
					return false;
				} // if
				if(!process_number(curr_token_.dvalue_)) {
					std::cerr << "error: could not process number '"
								<< curr_token_.dvalue_ << "'" << std::endl;
					return false;
				} // if
				break;

			case string_token:		// preceeded by '='
				if(past_token_.type_ != assignment_token &&
						past_token_.type_ != comment_token) {
					std::cerr << "error: stray string found '"
								<< curr_token_.svalue_ << "'" << std::endl;
					return false;
				} // if
				if(!process_string(curr_token_.svalue_)) {
					std::cerr << "error: could not process string "
								<< curr_token_.svalue_ << std::endl;
					return false;
				} // if
				break;

			case separator_token:	// should be preceeded by
									// array_end or string or number or object_end
				if(past_token_.type_ != array_end_token &&
						past_token_.type_ != object_end_token &&
						past_token_.type_ != string_token &&
						past_token_.type_ != number_token &&
						past_token_.type_ != comment_token) {
					std::cerr << "error: stray seperator token ',' found" << std::endl;
					return false;
				} // if
				break;

			default:				// this is for keyword tokens
									// read_oo_input makes sure there are no illegal tokens
									// this is always preceeded by ',' or '{'
				if(curr_token_.type_ != hiprmc_token &&
						past_token_.type_ != object_begin_token &&
						past_token_.type_ != separator_token &&
						past_token_.type_ != comment_token) {
					std::cerr << "error: keyword '" << curr_token_.svalue_
								<< "' not placed properly" << std::endl;
					return false;
				} // if
				past_keyword_ = curr_keyword_;
				curr_keyword_ = curr_token_.type_;
				if(!process_curr_keyword()) {
					std::cerr << "error: could not process current keyword '" << curr_token_.svalue_
								<< "'" << std::endl;
					return false;
				} // if
				break;
		} // switch

		return true;
	} // HipRMCInput::process_curr_token()


	bool HipRMCInput::process_curr_keyword() {
		// do some syntax error checkings
		switch(curr_keyword_) {

			case hiprmc_token:	// this will always be the first token
				if(past_token_.type_ != null_token || keyword_stack_.size() != 0) {
					std::cerr << "fatal error: 'hipRMCInput' token is not at the beginning!"
								<< std::endl;
					return false;
				} // if

				init();		// initialize everything
				break;

			case instrument_token:
			case instrument_input_image_token:
			case instrument_image_size_token:
			case instrument_num_tiles_token:
			case instrument_loading_token:
			case instrument_tstar_token:
			case instrument_cooling_factor_token:
			case compute_token:
			case compute_model_start_size_token:
			case compute_num_steps_factor_token:
			case compute_scale_factor_token:
			case compute_max_move_distance_token:
			case compute_label_token:
				break;

			default:
				std::cerr << "error: non-keyword token in keyword's position"
							<< std::endl;
				return false;
		} // switch()

		return true;
	} // HipRMCInput::process_curr_keyword()


	inline TokenType HipRMCInput::get_curr_parent() {
		if(keyword_stack_.size() < 1) return null_token;
		return keyword_stack_.top();
	} // HipRMCInput::get_curr_parent()


	inline TokenType HipRMCInput::get_curr_grandparent() {
		if(keyword_stack_.size() < 1) return null_token;
		TokenType temp = keyword_stack_.top();
		keyword_stack_.pop();
		if(keyword_stack_.size() < 1) { keyword_stack_.push(temp); return null_token; }
		TokenType gparent = keyword_stack_.top();
		keyword_stack_.push(temp);
		return gparent;
	} // HipRMCInput::get_curr_grandparent()


	bool HipRMCInput::process_number(const real_t& num) {
		TokenType parent = null_token;
		TokenType gparent = null_token;

		switch(curr_keyword_) {
			 
			case instrument_num_tiles_token:
				// find out which min is this for
				// shape param, scattering alphai, inplanerot, tilt
				parent = get_curr_parent();
				num_tiles_ = (unsigned int) num;
				break;

			case instrument_image_size_token:
				curr_vector_.push_back(num);
				if(curr_vector_.size() > 2) {
					std::cerr << "error: more than 2 values in imagesize" << std::endl;
					return false;
				} // if
				break;

			case instrument_loading_token:
				curr_vector_.push_back(num);
				break;

			case instrument_tstar_token:
				curr_vector_.push_back(num);
				break;

			case instrument_cooling_factor_token:
				curr_vector_.push_back(num);
				break;

			case compute_model_start_size_token:
				curr_vector_.push_back(num);
				if(curr_vector_.size() > 2) {
					std::cerr << "error: more than 2 values given in modelstartsize" << std::endl;
					return false;
				} // if
				break;

			case compute_num_steps_factor_token:
				num_steps_factor_ = (unsigned int) num;
				break;

			case compute_scale_factor_token:
				scale_factor_ = (unsigned int) num;
				break;

			case compute_max_move_distance_token:
				max_move_distance_ = (unsigned int) num;
				break;

			default:
				std::cerr << "fatal error: found a number '" << num
							<< "' where it should not be" << std::endl;
				return false;
		} // switch

		return true;
	} // HipRMCInput::process_number()


	bool HipRMCInput::process_string(const std::string& str) {
		TokenType parent = null_token;

		switch(curr_keyword_) {

			case instrument_input_image_token:
				input_image_ = str;
				break;

			case compute_label_token:
				label_ = str + "_" + timestamp();
				break;

			default:
				std::cerr << "fatal error: found a string '"
							<< str << "' where it should not be" << std::endl;
				return false;
		} // switch

		return true;
	} // HipRMCInput::process_string()



	/** print functions for testing only
	 */


	void HipRMCInput::print_all() {
		std::cout << "++ HipRMC Input data: " << std::endl;
		std::cout << "             Input image = " << input_image_ << std::endl;
		std::cout << "              Image size = " << image_size_[0] << " x " << image_size_[1] << std::endl;
		std::cout << "         Number of tiles = " << num_tiles_ << std::endl;
		std::cout << "         Loading factors = ";
		for(vec_real_t::iterator i = loading_factors_.begin(); i != loading_factors_.end(); ++ i)
			std::cout << *i << " ";
		std::cout << std::endl;
		std::cout << "            Temperatures = ";
		for(vec_real_t::iterator i = tstar_.begin(); i != tstar_.end(); ++ i)
			std::cout << *i << " ";
		std::cout << std::endl;
		std::cout << "         Cooling factors = ";
		for(vec_real_t::iterator i = cooling_factors_.begin(); i != cooling_factors_.end(); ++ i)
			std::cout << *i << " ";
		std::cout << std::endl;
		std::cout << "        Model start size = " << model_start_size_[0] << " x " << model_start_size_[1] << std::endl;
		std::cout << "  Number of steps factor = " << num_steps_factor_ << std::endl;
		std::cout << "          Scaling factor = " << scale_factor_ << std::endl;
		std::cout << "   Maximum move distance = " << max_move_distance_ << std::endl;
		std::cout << "                Run name = " << label_ << std::endl;
		std::cout << std::endl << std::flush;
	} // HipRMCInput::print_all()


} // namespace hir

