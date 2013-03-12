/***
  *  Project:
  *
  *  File: typedefs.hpp
  *  Created: Feb 03, 2013
  *  Modified: Sun 10 Mar 2013 10:57:57 AM PDT
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

typedef woo::Matrix2D <real_t>			mat_real_t;
typedef woo::Matrix2D <complex_t>		mat_complex_t;
typedef woo::Matrix2D <unsigned int>	mat_uint_t;

typedef std::vector <unsigned int>		vec_uint_t;

} // namespace hir

#endif // __TYPEDEFS_HPP__
