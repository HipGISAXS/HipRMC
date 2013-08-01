/***
  *  Project:
  *
  *  File: init_gpu.cuh
  *  Created: Feb 22, 2013
  *  Modified: Thu 01 Aug 2013 11:37:29 AM PDT
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef __INIT_GPU_CUH__
#define __INIT_GPU_CUH__

#include <cuda.h>
#include <cuda_runtime.h>

namespace hir {

	bool init_gpu() {
		std::cout << "-- Waking up GPU(s) ..." << std::flush << std::endl;
		cudaSetDevice(0);
		cudaFree(0);
		return true;
	} // init_gpu()

} // namespace hig

#endif // __INIT_GPU_CUH__
