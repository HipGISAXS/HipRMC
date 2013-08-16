/***
  *  Project:
  *
  *  File: tokens.hpp
  *  Created: Jun 11, 2013
  *  Modified: Thu 15 Aug 2013 09:55:24 PM PDT
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef __TOKENS_HPP__
#define __TOKENS_HPP__

#include "typedefs.hpp"

namespace hir {

	/**
	 * fundamental datatypes used
	 */

	enum ValueType {
		null_value = 0,					/* an empty/non-existant/null value */
		int_value,						/* integral number */
		real_value,						/* real number */
		string_value,					/* quoted strings */
		vector2_value,					/* a pair of two real numbers */
		vector3_value					/* a tuple of three real numbers */
	}; // enum ValueType


	/**
	 * set of valid tokens for an input component
	 */

	enum TokenType {
		/* sanity */
		null_token = 0,					/* when there is nothing yet */
		error_token,					/* there is something that is not supposed to be there */

		/* fundamental raw tokens: single character (they make bigger stuff) */
		character_token,				/* a single alphabet character */
		digit_token,					/* a single digit [0-9\-] */
		negative_token,					/* the negative sign '-' (for numbers only) */
		white_space_token,				/* a white space character: ' ' '\n' '\t' '\r' '\v' etc. */
		object_begin_token,				/* '{' */
		object_end_token,				/* '}' */
		array_begin_token,				/* '[' */
		array_end_token,				/* ']' */
		string_begin_end_token,			/* '"' */
		assignment_token,				/* '=' */
		separator_token,				/* ',' */
		comment_token,					/* sign '#' for a comment start */

		/**
		 * rig tokens: valid keywords and values
		 */

		hiprmc_token,					/* the main keyword representing the input object */

		/* value datatypes */
		number_token,					/* a number: integeral or real */
		string_token,					/* a quoted string */

		/* specific tokens */

		instrument_token,				/* the instrumentation object token */
		instrument_input_image_token,	/* the input image file token */
		instrument_image_size_token,	/* input image size token */
		instrument_num_tiles_token,		/* number of tiles token */
		instrument_loading_token,		/* loading factors array token */
		instrument_tstar_token,			/* temperature array token */
		instrument_cooling_factor_token,	/* cooling factors array token */

		compute_token,					/* computation object token */
		compute_model_start_size_token,	/* size of the starting model token */
		compute_num_steps_factor_token,	/* num steps factor token */
		compute_scale_factor_token,		/* scaling factor token */
		compute_max_move_distance_token,	/* max move distance token */
		compute_label_token				/* a label for output directory */
	}; // enum TokenType


	/**
	 * token class storing details of a token
	 */

	class Token {
		public:
			Token() : type_(null_token), svalue_(""), dvalue_(0.0) { }
			~Token() { }

			TokenType type_;		/* token type */
			std::string svalue_;	/* token's actual string value - if non-numeric */
			real_t dvalue_;		/* token's actual numeric value */
	}; // class Token

} // namespace hir

#endif /* __TOKENS_HPP__ */
