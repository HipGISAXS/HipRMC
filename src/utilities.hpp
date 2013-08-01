/**
 *  Project:
 *
 *  File: utilities.hpp
 *  Created: Aug 01, 2013
 *  Modified: Thu 01 Aug 2013 11:54:42 AM PDT
 *
 *  Author: Abhinav Sarje <asarje@lbl.gov>
 */

#ifndef __UTILITIES_HPP__
#define __UTILITIES_HPP__

namespace hir {

	/**
	 * returns a string with current timestamp
	 */
	std::string timestamp() {
		time_t rawtime;
		struct tm * timeinfo;
		char buffer[16];

		time(&rawtime);
		timeinfo = localtime(&rawtime);
		strftime(buffer, 16, "%Y%m%d_%H%M%S", timeinfo);

		return std::string(buffer);
	} // timestamp()

} // namespace

#endif // __UTILITIES_HPP_
