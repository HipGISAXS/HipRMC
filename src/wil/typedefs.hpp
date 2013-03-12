/***
  *  Project: HipGISAXS (High-Performance GISAXS)
  *
  *  File: typedefs.hpp
  *  Created: Jul 08, 2012
  *  Modified: Sat 09 Mar 2013 02:36:08 PM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef _TYPEDEFS_HPP_
#define _TYPEDEFS_HPP_

#include <vector>
#include <complex>
#ifdef USE_GPU 
#include <cuComplex.h>
#endif

namespace wil {

#ifdef DOUBLEP	// double precision
	typedef double						real_t;
	#ifdef USE_GPU
		typedef cuDoubleComplex			cucomplex_t;
	#endif
#else			// single precision
	typedef float						real_t;
	#ifdef USE_GPU
		typedef cuFloatComplex			cucomplex_t;
	#endif
#endif	// DOUBLEP

	typedef std::complex<real_t>		complex_t;

	typedef std::vector<real_t> 		real_vec_t;
	typedef std::vector<complex_t>		complex_vec_t;
#ifdef USE_GPU
	typedef std::vector<cucomplex_t>	cucomplex_vec_t;
#endif

} // namespace


#endif /* _TYPEDEFS_HPP_ */
