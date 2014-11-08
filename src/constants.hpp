/***
  *  Project: HipRMC
  *
  *  File: constants.hpp
  *  Created: Jan 28, 2013
  *  Modified: Fri 07 Nov 2014 12:03:32 PM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef __CONSTANTS_HPP__
#define __CONSTANTS_HPP__

namespace hir {

	const double ZERO_LIMIT_ = 1e-30;
	const double PI_ = 3.14159265358979323846;

	//const unsigned int MAX_NUM_PROCS = 2097152;
	const unsigned int MAX_NUM_PROCS = 512;
  const double MAX_TEMPERATURE = 3.0;

	#ifdef USE_GPU
		const unsigned int CUDA_BLOCK_SIZE_X_ = 16;
		const unsigned int CUDA_BLOCK_SIZE_Y_ = 16;

		const unsigned int CUDA_DFT2_SUBTILES_ = 4;
	#endif

} // namespace hir

#endif // __CONSTANTS_HPP__
