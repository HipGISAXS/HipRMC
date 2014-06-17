/***
  *  Project: WOO Random Number Generator Library
  *
  *  File: woo_mtrandom.hpp
  *  Created: Aug 25, 2013
  *  Modified: Sun 25 Aug 2013 01:55:38 PM PDT
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
			// construct with 0 as seed
			MTRandomNumberGenerator() {
				mt_rand_gen_.seed(0);
				min_ = mt_rand_gen_.min();
				max_ = mt_rand_gen_.max();
				last_ = -1.0;	// nothing
			} // MTRandomNumberGenerator()

			// construct with a given seed
			MTRandomNumberGenerator(unsigned int seed) {
				mt_rand_gen_.seed(seed);
				min_ = mt_rand_gen_.min();
				max_ = mt_rand_gen_.max();
				last_ = -1.0;	// nothing
			} // MTRandomNumberGenerator()

			~MTRandomNumberGenerator() { }

			void reset() {
				mt_rand_gen_.seed(0);
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
