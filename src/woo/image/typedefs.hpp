/***
  *  Project: WOO Image Library
  *
  *  File: typedefs.hpp
  *  Created: Jul 08, 2012
  *  Modified: Sun 25 Aug 2013 09:24:13 AM PDT
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef _TYPEDEFS_HPP_
#define _TYPEDEFS_HPP_

#include <vector>
#include <complex>

namespace wil {

#ifdef DOUBLEP	// double precision
	typedef double						real_t;
#else			// single precision
	typedef float						real_t;
#endif

	typedef std::complex<real_t>		complex_t;

	typedef std::vector<real_t> 		real_vec_t;
	typedef std::vector<complex_t>		complex_vec_t;

} // namespace


#endif /* _TYPEDEFS_HPP_ */
