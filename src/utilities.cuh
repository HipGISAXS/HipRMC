/***
  *  Project:
  *
  *  File: utilities.cuh
  *  Created: Feb 02, 2013
  *  Modified: Mon 04 Mar 2013 09:27:45 AM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef __UTILITIES_CUH__
#define __UTILITIES_CUH__

namespace hir {

	// overloaded arithmetic operations on cucomplex numbers

	// make complex number
	__host__ __device__
	cuFloatComplex make_cuComplex(float a, float b) { return make_cuFloatComplex(a, b); }
	__host__ __device__
	cuDoubleComplex make_cuComplex(double a, double b) { return make_cuDoubleComplex(a, b); }

	// add complex numbers
	__host__ __device__
	cuFloatComplex complex_add(cuFloatComplex a, cuFloatComplex b) { return cuCaddf(a, b); }
	__host__ __device__
	cuDoubleComplex complex_add(cuDoubleComplex a, cuDoubleComplex b) { return cuCadd(a, b); }

	// subtract complex numbers
	__host__ __device__
	cuFloatComplex complex_sub(cuFloatComplex a, cuFloatComplex b) { return cuCsubf(a, b); }
	__host__ __device__
	cuDoubleComplex complex_sub(cuDoubleComplex a, cuDoubleComplex b) { return cuCsub(a, b); }

	// multiply complex numbers
	__host__ __device__
	cuFloatComplex complex_mul(cuFloatComplex a, cuFloatComplex b) { return cuCmulf(a, b); }
	__host__ __device__
	cuDoubleComplex complex_mul(cuDoubleComplex a, cuDoubleComplex b) { return cuCmul(a, b); }

	// divide complex numbers
	__host__ __device__
	cuFloatComplex complex_div(cuFloatComplex a, cuFloatComplex b) { return cuCdivf(a, b); }
	__host__ __device__
	cuDoubleComplex complex_div(cuDoubleComplex a, cuDoubleComplex b) { return cuCdiv(a, b); }
	__host__ __device__
	cuFloatComplex complex_div(cuFloatComplex a, float b) { return make_cuFloatComplex(cuCrealf(a) / b, cuCimagf(a) / b); }
	__host__ __device__
	cuDoubleComplex complex_div(cuDoubleComplex a, double b) { return make_cuDoubleComplex(cuCreal(a) / b, cuCimag(a) / b); }

} // namespace hir

#endif // __UTILITIES_CUH__
