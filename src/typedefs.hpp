/***
  *  Project:
  *
  *  File: typedefs.hpp
  *  Created: Feb 03, 2013
  *  Modified: Wed 12 Jun 2013 01:28:16 PM PDT
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef __TYPEDEFS__HPP__
#define __TYPEDEFS__HPP__

#include <complex>
#ifndef __CUDACC__
#include <array>		// in c++11 standard
#endif
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

typedef std::vector <real_t>			vec_real_t;
typedef std::vector <unsigned int>		vec_uint_t;
//typedef std::vector <real_t>			vec2_real_t;	// ... replace with "array" fixed size TODO
//typedef std::vector <int>				vec2_int_t;		// ... ''
#ifndef __CUDACC__
typedef std::array <real_t, 2>			vec2_real_t;
typedef std::array <int, 2>				vec2_int_t;
typedef std::array <unsigned int, 2>	vec2_uint_t;
#endif

} // namespace hir

#endif // __TYPEDEFS_HPP__
