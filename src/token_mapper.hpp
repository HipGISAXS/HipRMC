/***
  *  Project:
  *
  *  File: token_mapper.hpp
  *  Created: Jun 11, 2013
  *  Modified: Tue 11 Jun 2013 03:42:09 PM PDT
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef __TOKEN_MAPPER_HPP__
#define __TOKEN_MAPPER_HPP__

#include <unordered_map>
#include <string>

#include "tokens.hpp"
#include "enums.hpp"

namespace hir {

	class TokenMapper {
		private:
			std::unordered_map <std::string, TokenType> KeyWords_;

		public:
			static TokenMapper& instance() {
				static TokenMapper token_mapper;
				return token_mapper;
			} // instance()

			
			TokenType get_keyword_token(const std::string& str) {
				if(KeyWords_.count(str) > 0) return KeyWords_[str];
				else return error_token;
			} // get_keyword_token()

			
			bool key_exists(const std::string& str) {
				if(KeyWords_.count(str) > 0) return true;
				return false;
			} // keyword_exists()


			bool keyword_token_exists(TokenType token) {
				// a sequential search for value 'token'
				std::unordered_map <std::string, TokenType>::iterator i = KeyWords_.begin();
				while(i != KeyWords_.end()) { if((*i).second == token) return true; i ++; }
				return false;
			} // token_exists()


		private:

			/* constructor */
			TokenMapper() {

				/* language keywords */

				KeyWords_[std::string("hipRMCInput")]		= hiprmc_token;
				KeyWords_[std::string("hiprmcinput")]		= hiprmc_token;
				KeyWords_[std::string("instrumentation")]	= instrument_token;
				KeyWords_[std::string("computation")]		= compute_token;
				KeyWords_[std::string("inputimage")]		= instrument_input_image_token;
				KeyWords_[std::string("imagesize")]			= instrument_image_size_token;
				KeyWords_[std::string("numtiles")]			= instrument_num_tiles_token;
				KeyWords_[std::string("loadingfactors")]	= instrument_loading_token;
				KeyWords_[std::string("modelstartsize")]	= compute_model_start_size_token;
				KeyWords_[std::string("numstepsfactor")]	= compute_num_steps_factor_token;
				KeyWords_[std::string("scalefactor")]		= compute_scale_factor_token;
			
		} // TokenMapper()

		// singleton
		TokenMapper(const TokenMapper&);
		TokenMapper& operator=(const TokenMapper&);

		bool has_extension(const std::string& s, const std::string& e) const {
			unsigned int p = s.rfind(e);
			if(p != s.length() - e.length())	// s is not a suffix of
				return false;
			return true;
		} // has_extension()

	}; // class TokenMapper

} // namespace hir

#endif /* __TOKEN_MAPPER_HPP__ */
