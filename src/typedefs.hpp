/***
  *  Project:
  *
  *  File: typedefs.hpp
  *  Created: Feb 03, 2013
  *  Modified: Mon 04 Mar 2013 09:18:58 AM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef __TYPEDEFS__HPP__
#define __TYPEDEFS__HPP__

#include <complex>
#ifdef USE_GPU
#include <cuComplex.h>
#endif // USE_GPU
#include <woo/matrix/matrix.hpp>

namespace hir {

#ifdef DOUBLEP	// double precision
	typedef double 				real_t;
	#ifdef USE_GPU
		typedef cuDoubleComplex	cucomplex_t;
	#endif
#else			// single precision
	typedef float 				real_t;
	#ifdef USE_GPU
		typedef cuFloatComplex	cucomplex_t;
	#endif
#endif // DOUBLEP

typedef std::complex <real_t>	complex_t;

typedef woo::Matrix2D <real_t>		mat_real_t;
typedef woo::Matrix2D <complex_t>	mat_complex_t;


} // namespace hir

#endif // __TYPEDEFS_HPP__
