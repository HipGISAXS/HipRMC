/***
  *  $Id$
  *
  *  Project:
  *
  *  File: woo_gtodtimers.hpp
  *  Created: Nov 21, 2012
  *  Modified: Sun 31 Mar 2013 03:48:36 PM PDT
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#include "wootimers.hpp"
#include <sys/time.h>

#ifndef _WOOGTODTIMERS_HPP_
#define _WOOGTODTIMERS_HPP_

namespace woo {

class GTODTimer : public WooTimer {
	// use system gettimeofday
	// lowest resolution is 1 microsecond (supposed to be)
	private:

	public:
		GTODTimer() { reset(); }
		~GTODTimer() { }

		void reset() { start_ = 0.0; stop_ = 0.0; elapsed_ = 0.0; is_running_ = false; }

		void start() {
			reset();
			struct timeval temptime;
			gettimeofday(&temptime, NULL);
			start_ = temptime.tv_sec * 1e6 + temptime.tv_usec;
			is_running_ = true;
		} // start()

		void stop() {
			if(!is_running_) {
				std::cerr << "error: timer is not running!" << std::endl;
				return;
			} // if
			struct timeval temptime;
			gettimeofday(&temptime, NULL);
			stop_ = temptime.tv_sec * 1e6 + temptime.tv_usec;
			is_running_ = false;
			elapsed_ = stop_ - start_;
		} // stop()

		double lap() {
			if(!is_running_) {
				std::cerr << "error: timer is not running!" << std::endl;
				return 0.0;
			} // if
			struct timeval temptime;
			gettimeofday(&temptime, NULL);
			double temp1 = temptime.tv_sec * 1e6 + temptime.tv_usec;
			double temp2 = temp1 - start_;
			elapsed_ += temp2;
			start_ = temp1;
			return temp2;
		} // lap()

		double elapsed_sec() { return elapsed_ / 1e6; }
		double elapsed_msec() { return elapsed_ / 1e3; }
		double elapsed_usec() { return elapsed_; }
		double elapsed_nsec() { return elapsed_ * 1e3; }
}; // class WooGTODTimer

} // namespace woo


#endif // _WOOGTODTIMERS_HPP_
