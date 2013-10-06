/***
  *  $Id$
  *
  *  Project:
  *
  *  File: woo_cxxtimers.hpp
  *  Created: Nov 21, 2012
  *  Modified: Sun 31 Mar 2013 02:04:12 PM PDT
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#include "wootimers.hpp"
#include <ctime>

#ifndef _WOOCXXTIMERS_HPP_
#define _WOOCXXTIMERS_HPP_

namespace woo {

class CXXTimer : public WooTimer {
	// use c++ ctime: time(), clock() etc.
	// lowest resolution is 1 CPU clock tick
	private:
		clock_t clock_start_;
		clock_t clock_stop_;

	public:
		CXXTimer() { reset(); }
		~CXXTimer() { }

		void reset() { start_ = 0.0; stop_ = 0.0; elapsed_ = 0.0; is_running_ = false; }

		void start() {
			reset();
			clock_start_ = clock();
			start_ = ((double) clock_start_) / CLOCKS_PER_SEC;
			is_running_ = true;
		} // start()

		void stop() {
			if(!is_running_) {
				std::cerr << "error: timer is not running!" << std::endl;
				return;
			} // if
			clock_stop_ = clock();
			stop_ = ((double) clock_stop_) / CLOCKS_PER_SEC;
			is_running_ = false;
			elapsed_ = ((double) clock_stop_ - (double) clock_start_) / CLOCKS_PER_SEC;
		} // stop()

		double lap() {
			if(!is_running_) {
				std::cerr << "error: timer is not running!" << std::endl;
				return 0.0;
			} // if
			clock_t temp1 = clock();
			double temp2 = ((double) temp1 - (double) clock_start_) / CLOCKS_PER_SEC;
			elapsed_ += temp2;
			clock_start_ = temp1;
			start_ = ((double) temp1) / CLOCKS_PER_SEC;
			return temp2;
		} // lap()

		double elapsed_sec() { return elapsed_; }
		double elapsed_msec() { return elapsed_ * 1e3; }
		double elapsed_usec() { return elapsed_ * 1e6; }
		double elapsed_nsec() { return elapsed_ * 1e9; }
}; // class WooCppTimer

} // namespace woo


#endif // _WOOCXXTIMERS_HPP_
