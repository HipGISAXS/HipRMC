/***
  *  Project: HipRMC
  *
  *  File: constants.hpp
  *  Created: Jan 28, 2013
  *  Modified: Mon 18 Mar 2013 10:38:38 AM PDT
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef __CONSTANTS_HPP__
#define __CONSTANTS_HPP__

namespace hir {

	const double ZERO_LIMIT_ = 0.0000001;
	const double PI_ = 3.14159265358979323846;

	#ifdef USE_GPU
		const unsigned int CUDA_BLOCK_SIZE_X_ = 16;
		const unsigned int CUDA_BLOCK_SIZE_Y_ = 16;

		const unsigned int CUDA_DFT2_SUBTILES_ = 4;
	#endif

} // namespace hir

#endif // __CONSTANTS_HPP__
