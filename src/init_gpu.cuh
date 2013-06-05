/***
  *  Project:
  *
  *  File: init_gpu.cuh
  *  Created: Feb 22, 2013
  *  Modified: Wed 05 Jun 2013 12:34:19 PM PDT
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
		//cudaSetDevice(2);
		cudaFree(0);
		return true;
	} // init_gpu()

} // namespace hig

#endif // __INIT_GPU_CUH__
