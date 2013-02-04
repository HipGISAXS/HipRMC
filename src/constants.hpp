/***
  *  Project: HipRMC
  *
  *  File: constants.hpp
  *  Created: Jan 28, 2013
  *  Modified: Sat 02 Feb 2013 05:12:30 PM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef __CONSTANTS_HPP__
#define __CONSTANTS_HPP__

namespace hir {

	const double ZERO_LIMIT_ = 0.0000001;
	const double PI_ = 3.14159265358979323846;

	const unsigned int CUDA_BLOCK_SIZE_X_ = 8;
	const unsigned int CUDA_BLOCK_SIZE_Y_ = 8;

} // namespace hir

#endif // __CONSTANTS_HPP__
