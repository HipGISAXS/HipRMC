/***
  *  Project:
  *
  *  File: init_gpu.cuh
  *  Created: Feb 22, 2013
  *  Modified: Fri 08 Mar 2013 08:50:46 PM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef __INIT_GPU_CUH__
#define __INIT_GPU_CUH__

#include <cuda.h>
#include <cuda_runtime.h>

namespace hir {

	void init_gpu() {
		std::cout << "-- Waking up GPU(s) ..." << std::flush << std::endl;
		//cudaSetDevice(2);
		cudaFree(0);
	} // init_gpu()

} // namespace hig

#endif // __INIT_GPU_CUH__
