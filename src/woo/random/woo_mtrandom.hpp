/***
  *  Project: WOO Random Number Generator Library
  *
  *  File: woo_mtrandom.hpp
  *  Created: Aug 25, 2013
  *  Modified: Thu 13 Nov 2014 07:20:39 PM EST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef __WOO_RANDOM_MT_HPP__
#define __WOO_RANDOM_MT_HPP__

#include "woorandomnumbers.hpp"
#include <random>

namespace woo {

	// C++ std Mersenne-Twister random number generator
	class MTRandomNumberGenerator : public WooRandomNumberGenerator {
		private:
			// random number generator
			std::mt19937_64 mt_rand_gen_;

			// return a random number in (0,1)
			double mt_rand_01() {
				return ((double) (mt_rand_gen_() - min_) / (max_ - min_));
			} // mt_rand_01()

		public:
			// default is time as seed
			MTRandomNumberGenerator() {
				min_ = mt_rand_gen_.min();
				max_ = mt_rand_gen_.max();
				last_ = -1.0;	// nothing
			} // MTRandomNumberGenerator()

			// construct with a given seed
			MTRandomNumberGenerator(unsigned int seed):
				mt_rand_gen_(seed) {
				min_ = mt_rand_gen_.min();
				max_ = mt_rand_gen_.max();
				last_ = -1.0;	// nothing
			} // MTRandomNumberGenerator()

			~MTRandomNumberGenerator() { }

			void reset() {
				mt_rand_gen_.seed(time(NULL));
				last_ = -1.0;
			} // reset()

			void reset(unsigned int seed) {
				mt_rand_gen_.seed(seed);
				last_ = -1.0;
			} // reset()

			//double min() { return min_; }

			//double max() { return max_; }

			// returns the next random number
			double rand() {
				last_ = mt_rand_01();
				return last_;
			} // rand()

			double rand_last() { return last_; }
	}; // class WooRandomNumberGenerator

} // namespace woo

#endif // __WOO_RANDOM_MT_HPP__
