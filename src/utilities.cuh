/***
  *  Project:
  *
  *  File: utilities.cuh
  *  Created: Feb 02, 2013
  *  Modified: Sun 03 Feb 2013 01:08:41 PM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef __UTILITIES_CUH__
#define __UTILITIES_CUH__

namespace hir {

	__host__ __device__
	cuFloatComplex make_cuComplex(float a, float b) { return make_cuFloatComplex(a, b); }
	__host__ __device__
	cuDoubleComplex make_cuComplex(double a, double b) { return make_cuDoubleComplex(a, b); }

} // namespace hir

#endif // __UTILITIES_CUH__
