/***
  *  Project:
  *
  *  File: typedefs.hpp
  *  Created: Feb 03, 2013
  *  Modified: Sun 03 Feb 2013 12:13:12 PM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef __TYPEDEFS__HPP__
#define __TYPEDEFS__HPP__

#include <complex>
#ifdef USE_GPU
#include <cuComplex.h>
#endif // USE_GPU

namespace hir {

#ifdef DOUBLEP	// double precision
	typedef double 					real_t;
	typedef std::complex<double>	complex_t;
#ifdef USE_GPU
	typedef cuDoubleComplex			cucomplex_t;
#endif // USE_GPU
#else			// single precision
	typedef float 					real_t;
	typedef std::complex<float>		complex_t;
#ifdef USE_GPU
	typedef cuFloatComplex			cucomplex_t;
#endif // USE_GPU
#endif // DOUBLEP

} // namespace hir

#endif // __TYPEDEFS_HPP__
