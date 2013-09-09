/**
 *  Project:
 *
 *  File: lw_solver.hpp
 *  Created: Sep 06, 2013
 *  Modified: Fri 06 Sep 2013 06:42:27 PM PDT
 *
 *  Author: Abhinav Sarje <asarje@lbl.gov>
 */

#ifndef __LW_SOLVER_HPP__
#define __LW_SOLVER_HPP__

#include <map>
#include <gsl/gsl_multifit_nlin.h>

#include "typedefs.hpp"

namespace hir {

	typedef std::map <const real_t, real_t> map_real_t;

	int sigmoid_f(const gsl_vector *params, void *data, gsl_vector *f) {
		size_t n = (*(map_real_t*)data).size();

		double a = gsl_vector_get(params, 0);
		double b = gsl_vector_get(params, 1);

		int i = 0;
		for(map_real_t::const_iterator di = (*(map_real_t*)data).begin();
				di != (*(map_real_t*)data).end(); ++ di, ++ i) {
			// model sigmoid function yi = 1 / (1 + e^-((xi - a) / b))
			real_t yi = 1.0 / (1.0 + exp((a - (*di).first) / b));
			gsl_vector_set(f, i, yi - (*di).second);
		} // for

		return GSL_SUCCESS;
	} // sigmoid_f()


	int sigmoid_df(const gsl_vector *params, void *data, gsl_matrix *J) {
		size_t n = (*(map_real_t*)data).size();

		double a = gsl_vector_get(params, 0);
		double b = gsl_vector_get(params, 1);

		int i = 0;
		for(map_real_t::const_iterator di = (*(map_real_t*)data).begin();
				di != (*(map_real_t*)data).end(); ++ di, ++ i) {
			// Jacobian matrix J[i,j] = d fi / d pj
			// fi = (Yi - yi)
			// Yi = 1 / (1 + e^-((xi - a) / b))
			// pj are the paramaters (a, b)
			real_t x = (*di).first;
			real_t t1 = (x - a) / b;
			real_t t2 = exp(t1);
			real_t t3 = 1.0 / t2;
			real_t j1 = - 1.0 / (b * (2.0 + t2 + t3));
			real_t j2 = - (t1 / b) * (1.0 / (2.0 + t2 + t3));
			gsl_matrix_set(J, i, 0, j1);
			gsl_matrix_set(J, i, 1, j2);
		} // for

		return GSL_SUCCESS;
	} // sigmoid_f()


	int sigmoid_fdf(const gsl_vector *params, void *data, gsl_vector *f, gsl_matrix *J) {
		sigmoid_f(params, data, f);
		sigmoid_df(params, data, J);
		return GSL_SUCCESS;
	} // sigmoid_fdf


	/**
	 * Levenberg-Marquardt solver using GSL
	 */
	class LWSolver {	// currently implements sigmoid function only

		private:

		public:
			LWSolver() { }
			~LWSolver() { }

			bool solve_sigmoid(std::map<const real_t, real_t>& data, real_t& a, real_t& b) {
				const size_t n = data.size();

				const size_t p = 2;
				real_t params[p] = { a, b };
				gsl_vector_view param_vec = gsl_vector_view_array(params, p);

				gsl_multifit_function_fdf f;
				f.f = &sigmoid_f;
				f.df = &sigmoid_df;
				f.fdf = &sigmoid_fdf;
				f.n = n;
				f.p = p;
				f.params = &data;

				const gsl_multifit_fdfsolver_type* T = gsl_multifit_fdfsolver_lmsder;
				gsl_multifit_fdfsolver* s = gsl_multifit_fdfsolver_alloc(T, n, p);
				gsl_multifit_fdfsolver_set(s, &f, &param_vec.vector);

				int status = GSL_CONTINUE;
				for(int i = 0; i < 500 && status == GSL_CONTINUE; ++ i) {
					status = gsl_multifit_fdfsolver_iterate(s);
					if(status) break;
					status = gsl_multifit_test_delta(s->dx, s->x, 1e-4, 1e-4);
				} // for

				a = gsl_vector_get(s->x, 0);
				b = gsl_vector_get(s->x, 1);

				gsl_multifit_fdfsolver_free(s);
				return true;
			} // solve_sigmoid()
	}; // class LWSolver

} // namespace hir

#endif // __LW_SOLVER_HPP__
