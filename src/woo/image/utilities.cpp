/***
  *  Project: WOO Image Library
  *
  *  File: utilities.cpp
  *  Created: Jun 25, 2012
  *  Modified: Sun 25 Aug 2013 09:24:14 AM PDT
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#include <iostream>
#include <cmath>
#include <boost/math/special_functions/fpclassify.hpp>

#include "utilities.hpp"

namespace wil {


	/**
	 * specialization for vector3_t
	 */
	template <>
	vector3_t min <vector3_t> (vector3_t a, vector3_t b) {
		return vector3_t((a[0] < b[0] ? a[0] : b[0]), (a[1] < b[1] ? a[1] : b[1]),
						(a[2] < b[2] ? a[2] : b[2]));
	} // min <vector3_t>

	/**
	 * specialization for vector3_t
	 */
	template <>
	vector3_t max <vector3_t> (vector3_t a, vector3_t b) {
		return vector3_t((a[0] > b[0] ? a[0] : b[0]), (a[1] > b[1] ? a[1] : b[1]),
						(a[2] > b[2] ? a[2] : b[2]));
	} // max <vector3_t>


	/**
	 * apply log10 to all elements of the 2D matrix
	 */
	bool mat_log10_2d(unsigned int x_size, unsigned int y_size, real_t* &data) {
		if(data == NULL) {
			std::cerr << "error: data is null while calculating log10" << std::endl;
			return false;
		} // if
		for(unsigned int i = 0; i < x_size * y_size; ++ i) {
			if(data[i] <= 0) {
				if(data[i] == 0) {
					data[i] = 0.0;
					continue;
				} else {
					std::cerr << "error: matrix has a negative value. cannot calculate logarithm"
							<< std::endl;
					return false;
				} // if-else
			} else
				data[i] = log10(data[i]);
		} // for
		return true;
	} // mat_log10()


	/**
	 * compute the transpose of a matrix
	 * use boost libs ...
	 */
	bool transpose(unsigned int x_size, unsigned int y_size, const real_t *matrix, real_t* &transp) {
		if(matrix == NULL) {
			std::cerr << "error: matrix is NULL while tranposing" << std::endl;
			return false;
		} // if
		transp = new (std::nothrow) real_t[x_size * y_size];
		for(unsigned int y = 0; y < y_size; ++ y) {
			for(unsigned int x = 0; x < x_size; ++ x) {
				transp[y_size * x + y] = matrix[x_size * y + x];
			} // for x
		} // for y

		return true;
	} // transpose()


	/**
	 * matrix multiplication for two 3x3 matrices
	 * operation is:
	 * x1 x2 x3   a1 a2 a3   d1 d2 d3
	 * y1 y2 y3 = b1 b2 b3 x e1 e2 e3
	 * z1 z2 z3   c1 c2 c3   f1 f2 f3
	 * use boost libs ... and make it general ...
	 */
	bool mat_mul_3x3(vector3_t a, vector3_t b, vector3_t c,
					vector3_t d, vector3_t e, vector3_t f,
					vector3_t& x, vector3_t& y, vector3_t& z) {
		real_t *A = new (std::nothrow) real_t[9];
		real_t *B = new (std::nothrow) real_t[9];
		real_t *C = new (std::nothrow) real_t[9];

		A[0] = a[0]; A[1] = a[1]; A[2] = a[2];
		A[3] = b[0]; A[4] = b[1]; A[5] = b[2];
		A[6] = c[0]; A[7] = c[1]; A[8] = c[2];
		B[0] = d[0]; B[1] = d[1]; B[2] = d[2];
		B[3] = e[0]; B[4] = e[1]; B[5] = e[2];
		B[6] = f[0]; B[7] = f[1]; B[8] = f[2];

		for(int i = 0; i < 3; i ++) {
			for(int j = 0; j < 3; j ++) {
				C[3 * i + j] = 0.0;
				for(int k = 0; k < 3; k ++) {
					C[3 * i + j] += A[3 * i + k] * B[3 * k + j];
				} // for k
			} // for j
		} // for i

		x[0] = C[0]; x[1] = C[1]; x[2] = C[2];
		y[0] = C[3]; y[1] = C[4]; y[2] = C[5];
		z[0] = C[6]; z[1] = C[7]; z[2] = C[8];

		delete[] C;
		delete[] B;
		delete[] A;
		return true;
	} // mat_mul_3x3()


	/**
	 * matrix vector product for matrix of size 3x3 and vector of size 1x3
	 * operation is:
	 * x1   a1 a2 a3   d1
	 * x2 = b1 b2 b3 x d2
	 * x3   c1 c2 c3   d3
	 * note: transpose of d is used
	 * use boost libs ...
	 */
	bool mat_mul_3x1(vector3_t a, vector3_t b, vector3_t c, vector3_t d, vector3_t& x) {
		x[0] = a[0] * d[0] + a[1] * d[1] + a[2] * d[2];
		x[1] = b[0] * d[0] + b[1] * d[1] + b[2] * d[2];
		x[2] = c[0] * d[0] + c[1] * d[1] + c[2] * d[2];

		return true;
	} // mat_mul_3x1()


	/**
	 * specialized floor function
	 */
	vector3_t floor(vector3_t a) {
		return vector3_t(std::floor(a[0]), std::floor(a[1]), std::floor(a[2]));
	} // floor()


	/**
	 * comparison of complex numbers
	 */
	bool operator<(complex_t a, complex_t b) {
		if(std::real(a) < std::real(b)) return true;
		if(std::real(a) == std::real(b) && std::imag(a) < std::imag(b)) return true;
		//if(a.x < b.x) return true;
		//if(a.x == b.x && a.y < b.y) return true;
		return false;
	} // operator<()


	/**
	 * arithmetic operators for complex types
	 */

	complex_t operator*(complex_t c, complex_t s) {
		return complex_t(c.real() * s.real() - c.imag() * s.imag(),
						 c.real() * s.imag() + c.imag() * s.real());
	} // operator*()

	complex_t operator*(complex_t c, real_t s) {
		return complex_t(c.real() * s, c.imag() * s);
	} // operator*()

	complex_t operator*(real_t s, complex_t c) {
		return complex_t(c.real() * s, c.imag() * s);
	} // operator*()

	std::complex<long double> operator*(std::complex<long double> c, long double s) {
		return std::complex<long double>(c.real() * s, c.imag() * s);
	} // operator*()

	std::complex<long double> operator/(std::complex<long double> c, long double s) {
		return std::complex<long double>(c.real() / s, c.imag() / s);
	} // operator*()


	/**
	 * constructs element-by-element sum of two matrices into result
	 */
	bool mat_add(unsigned int x1_size, unsigned int y1_size, unsigned int z1_size,
					const std::vector<complex_t>& matrix1,
					unsigned int x2_size, unsigned int y2_size, unsigned int z2_size,
					const std::vector<complex_t>& matrix2,
					std::vector<complex_t>& result) {
		if(x1_size != x2_size || y1_size != y2_size || z1_size != z2_size
				|| matrix1.size() != matrix2.size()) {
			std::cerr << "error: matrix sizes are not the same for addition operation" << std::endl;
			return false;
		} // if
		result.clear();
		std::vector<complex_t>::const_iterator i1 = matrix1.begin();
		std::vector<complex_t>::const_iterator i2 = matrix2.begin();
		for(; i1 != matrix1.end(); ++ i1, ++ i2) {
			result.push_back((*i1) + (*i2));
		} // for
		return true;
	} // mat_add()

	/**
	 * performs in-place element-by-element sum of two matrices into first matrix
	 */
	bool mat_add_in(unsigned int x1_size, unsigned int y1_size, unsigned int z1_size,
					std::vector<complex_t>& matrix1,
					unsigned int x2_size, unsigned int y2_size, unsigned int z2_size,
					std::vector<complex_t>& matrix2) {
		if(x1_size != x2_size || y1_size != y2_size || z1_size != z2_size
				|| matrix1.size() != matrix2.size()) {
			std::cerr << "error: matrix sizes are not the same for addition operation" << std::endl;
			return false;
		} // if
		std::vector<complex_t>::iterator i1 = matrix1.begin();
		std::vector<complex_t>::iterator i2 = matrix2.begin();
		for(; i1 != matrix1.end(); ++ i1, ++ i2) {
			*i1 = (*i1) + (*i2);
		} // for
		return true;
	} // mat_add()


	/**
	 * scalar-matrix multiplication into result
	 */
	bool mat_mul(real_t scalar, const complex_vec_t& matrix, complex_vec_t& result) {
		result.clear();
		for(std::vector<complex_t>::const_iterator i = matrix.begin(); i != matrix.end(); ++ i) {
			result.push_back((*i) * scalar);
		} // for
		return true;
	} // mat_mul()

	bool mat_mul(complex_t scalar, const complex_vec_t& matrix, complex_vec_t& result) {
		result.clear();
		for(complex_vec_t::const_iterator i = matrix.begin(); i != matrix.end(); ++ i) {
			result.push_back((*i) * scalar);
		} // for
		return true;
	} // mat_mul()

	bool mat_mul(const complex_vec_t& matrix, real_t scalar, complex_vec_t& result) {
		return mat_mul(scalar, matrix, result);
	} // mat_mul()

	bool mat_mul(const complex_vec_t& matrix, complex_t scalar, complex_vec_t& result) {
		return mat_mul(scalar, matrix, result);
	} // mat_mul()

	/**
	 * in-place scalar-matrix multiplication
	 */
	bool mat_mul_in(real_t scalar,	complex_vec_t& matrix) {
		for(complex_vec_t::iterator i = matrix.begin(); i != matrix.end(); ++ i) {
			*i = (*i) * scalar;
		} // for
		return true;
	} // mat_mul()

	bool mat_mul_in(complex_t scalar, complex_vec_t& matrix) {
		for(complex_vec_t::iterator i = matrix.begin(); i != matrix.end(); ++ i) {
			*i = (*i) * scalar;
		} // for
		return true;
	} // mat_mul()

	bool mat_mul_in(complex_vec_t& matrix, real_t scalar) {
		return mat_mul_in(scalar, matrix);
	} // mat_mul()

	bool mat_mul_in(complex_vec_t& matrix, complex_t scalar) {
		return mat_mul_in(scalar, matrix);
	} // mat_mul()


	/**
	 * computes element-by-element product of two matrices into result
	 */
	bool mat_dot_prod(unsigned int x1_size, unsigned int y1_size, unsigned int z1_size,
						const complex_vec_t& matrix1,
						unsigned int x2_size, unsigned int y2_size, unsigned int z2_size,
						const complex_vec_t& matrix2,
						complex_vec_t& result) {
		if(x1_size != x2_size || y1_size != y2_size || z1_size != z2_size
				|| matrix1.size() != matrix2.size()) {
			std::cerr << "error: matrix sizes are not the same for dot product operation" << std::endl;
			return false;
		} // if
		result.clear();
		complex_vec_t::const_iterator i1 = matrix1.begin();
		complex_vec_t::const_iterator i2 = matrix2.begin();
		for(; i1 != matrix1.end(); ++ i1, ++ i2) {
			result.push_back((*i1) * (*i2));
		} // for
		return true;
	} // mat_dot_prod()

	/**
	 * performs in-place element-by-element product of two matrices
	 */
	bool mat_dot_prod_in(unsigned int x1_size, unsigned int y1_size, unsigned int z1_size,
							std::vector<complex_t>& matrix1,
							unsigned int x2_size, unsigned int y2_size, unsigned int z2_size,
							std::vector<complex_t>& matrix2) {
		if(x1_size != x2_size || y1_size != y2_size || z1_size != z2_size
				|| matrix1.size() != matrix2.size()) {
			std::cerr << "error: matrix sizes are not the same for dot product operation" << std::endl;
			return false;
		} // if
		std::vector<complex_t>::iterator i1 = matrix1.begin();
		std::vector<complex_t>::iterator i2 = matrix2.begin();
		for(; i1 != matrix1.end(); ++ i1, ++ i2) {
			*i1 = (*i1) * (*i2);
		} // for
		return true;
	} // mat_dot_prod()


	/**
	 * computes element-by-element division of two matrices (matrix1 / matrix2)	into result
	 */
	bool mat_dot_div(unsigned int nx1, unsigned int ny1, unsigned int nz1,
						const complex_vec_t& matrix1,
						unsigned int nx2, unsigned int ny2, unsigned int nz2,
						const complex_vec_t& matrix2,
						complex_vec_t& result) {
		if(nx1 != nx2 || ny1 != ny2 || nz1 != nz2 || matrix1.size() != matrix2.size()) {
			std::cerr << "error: matrix sizes are not the same for dot division operation"
						<< std::endl;
			return false;
		} // if
		result.clear();
		complex_vec_t::const_iterator i1 = matrix1.begin();
		complex_vec_t::const_iterator i2 = matrix2.begin();
		for(; i1 != matrix1.end(); ++ i1, ++ i2) {
			result.push_back((*i1) / (*i2));
		} // for
		return true;
	} // mat_dot_div()

	/**
	 * performs in-place element-by-element division of two matrices (matrix1/matrix2) into matrix1
	 */
	bool mat_dot_div_in(unsigned int nx1, unsigned int ny1, unsigned int nz1,
						std::vector<complex_t>& matrix1,
						unsigned int nx2, unsigned int ny2, unsigned int nz2,
						std::vector<complex_t>& matrix2) {
		if(nx1 != nx2 || ny1 != ny2 || nz1 != nz2 || matrix1.size() != matrix2.size()) {
			std::cerr << "error: matrix sizes are not the same for dot division operation"
						<< std::endl;
			return false;
		} // if
		std::vector<complex_t>::iterator i1 = matrix1.begin();
		std::vector<complex_t>::iterator i2 = matrix2.begin();
		for(; i1 != matrix1.end(); ++ i1, ++ i2) {
			*i1 = (*i1) / (*i2);
		} // for
		return true;
	} // mat_dot_div()


	bool mat_sqr(const complex_vec_t& matrix, complex_vec_t& result) {
		result.clear();
		for(complex_vec_t::const_iterator i = matrix.begin(); i != matrix.end(); ++ i) {
			result.push_back((*i) * (*i));
		} // for
		return true;
	} // mat_sqr()

	bool mat_sqr_in(//unsigned int nx, unsigned int ny, unsigned int nz,
						std::vector<complex_t>& matrix) {
		for(std::vector<complex_t>::iterator i = matrix.begin(); i != matrix.end(); ++ i) {
			*i = (*i) * (*i);
		} // for
		return true;
	} // mat_sqr()


	bool mat_sqrt(const complex_vec_t& matrix, complex_vec_t& result) {
		result.clear();
		for(complex_vec_t::const_iterator i = matrix.begin(); i != matrix.end(); ++ i) {
			result.push_back(sqrt(*i));
		} // for
		return true;
	} // mat_sqrt()

	bool mat_sqrt_in(//unsigned int nx, unsigned int ny, unsigned int nz,
						std::vector<complex_t>& matrix) {
		for(std::vector<complex_t>::iterator i = matrix.begin(); i != matrix.end(); ++ i) {
			*i = sqrt(*i);
		} // for
		return true;
	} // mat_sqrt()


	bool mat_exp(complex_vec_t& matrix, complex_vec_t& result) {
		result.clear();
		for(complex_vec_t::iterator i = matrix.begin(); i != matrix.end(); ++ i)
			result.push_back(exp(*i));
	} // mat_exp()


	bool mat_exp_in(complex_vec_t& matrix) {
		for(complex_vec_t::iterator i = matrix.begin(); i != matrix.end(); ++ i) *i = exp(*i);
	} // mat_exp_in()


	// compute integral of e^(ikx) between x1 and x2
	complex_t integral_e(real_t x1, real_t x2, complex_t k) {
		if(boost::math::fpclassify(k.real()) == FP_ZERO &&
				boost::math::fpclassify(k.imag()) == FP_ZERO) {
			return complex_t(x2 - x1, 0);
		} else {
			complex_t ik = complex_t(0.0, 1.0) * k;
			return (((real_t) 1.0 / ik) * (exp(ik * x2) - exp(ik * x1)));
		} // if-else
	} // integral_e


	// compute integral of (ax + b) e^(ikx) between x1 and x2
	complex_t integral_xe(real_t x1, real_t x2, real_t a, real_t b, complex_t k) {
		if(boost::math::fpclassify(k.real()) == FP_ZERO &&
				boost::math::fpclassify(k.imag()) == FP_ZERO) {
			return complex_t(a * (x2 * x2 - x1 * x1) / 2 + b * (x2 - x1), 0.0);
		} else {
			complex_t ik = complex_t(0.0, 1.0) * k;
			return (((real_t) 1.0 / ik) * ((a * x2 + b - a / ik) * exp(ik * x2) -
											(a * x1 + b - a / ik) * exp(ik * x1)));
		} // if-else
	} // integral_xe()

} // namespace wil
