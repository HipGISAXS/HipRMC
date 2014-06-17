/**
 *  Project: WOO Random Number Generator Library
 *
 *  File: woorandomnumbers.hpp
 *  Created: Aug 25, 2013
 *  Modified: Sun 25 Aug 2013 01:53:52 PM PDT
 *
 *  Author: Abhinav Sarje <asarje@lbl.gov>
 */

#ifndef __WOO_RANDOM_NUMBERS__
#define __WOO_RANDOM_NUMBERS__

namespace woo {

	class WooRandomNumberGenerator {	// an abstract class
		protected:
			double min_;	// the minimum value for a given random number generator
			double max_;	// the maximum value for a given random number generator

			double last_;	// stores the last generated random number

		public:
			virtual ~WooRandomNumberGenerator() { }

			virtual void reset() = 0;				// reset the random number generator
			virtual void reset(unsigned int) = 0;	// reset the random number generator with seed

			//virtual double min() = 0;					// return the min number
			//virtual double max() = 0;					// return the max number

			virtual double rand() = 0;				// returns a random number in [0,1]
			virtual double rand_last() = 0;			// returns the last generated random number
	}; // class WooRandomNumberGenerator

} // namespace woo

#endif // __WOO_RANDOM_NUMBERS__
