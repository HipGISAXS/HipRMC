/***
  *  $Id$
  *
  *  Project:
  *
  *  File: woo_cudatimers.hpp
  *  Created: Nov 21, 2012
  *  Modified: Thu 22 Nov 2012 10:54:52 AM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#include "wootimers.hpp"

#ifndef _WOOCUDATIMERS_HPP_
#define _WOOCUDATIMERS_HPP_

namespace woo {

class CUDATimer : public WooTimer {	// this does not use the start_ and stop_ of base class!!!
	// use CUDA timers
	// lowest resolution is msec (check with the CUDA benchmarks ... )
	private:
		cudaEvent_t custart_;
		cudaEvent_t custop_;
		cudaEvent_t cubase_;

		void elapsed_compute() {
			float temp1 = 0.0, temp2 = 0.0, temp3 = 0.0;
			cudaEventSynchronize(custop_);
			cudaEventElapsedTime(&temp1, cubase_, custart_);
			start_ = temp1;
			cudaEventElapsedTime(&temp2, cubase_, custop_);
			stop_ = temp2;
			cudaEventElapsedTime(&temp3, custart_, custop_);
			is_running_ = false;
			elapsed_ = temp3;	// same as stop_ - start_
		} // elapsed_compute()

	public:
		CUDATimer() {
			start_ = 0.0; stop_ = 0.0; elapsed_ = 0.0; is_running_ = false;
			cudaEventCreate(&custart_);
			cudaEventCreate(&custop_);
			cudaEventCreate(&cubase_);

			cudaEventRecord(cubase_);
		} // CUDATimer()

		~CUDATimer() {
			cudaEventDestroy(cubase_);
			cudaEventDestroy(custop_);
			cudaEventDestroy(custart_);
		} // ~CUDATimer()

		void reset() { start_ = 0.0; stop_ = 0.0; elapsed_ = 0.0; is_running_ = false; }

		void start() {
			reset();
			cudaEventRecord(custart_);
			is_running_ = true;
		} // start()

		void stop() {
			if(!is_running_) {
				std::cerr << "error: timer is not running" << std::endl;
				return;
			} // if
			cudaEventRecord(custop_);
		} // stop()

		double lap() {
			// not functional!
			return 0.0;
		} // lap()

		double elapsed_sec() {
			if(is_running_) elapsed_compute();
			return elapsed_ / 1e3;
		} // elapsed_sec()

		double elapsed_msec() {
			if(is_running_) elapsed_compute();
			return elapsed_;
		} // elapsed_msec()

		double elapsed_usec() {
			if(is_running_) elapsed_compute();
			return elapsed_ * 1e3;
		} // elapsed_usec()

		double elapsed_nsec() {
			if(is_running_) elapsed_compute();
			return elapsed_ * 1e6;
		} // elapsed_nsec()

}; // class WooCudaTimer

} // namespace woo


#endif // _WOOCUDATIMERS_HPP_
