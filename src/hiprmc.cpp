/***
  *  Project: HipRMC
  *
  *  File: hiprmc.cpp
  *  Created: Jan 27, 2013
  *  Modified: Fri 13 Sep 2013 09:21:06 AM PDT
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */


#include <iostream>

#include "rmc.hpp"

int main(int narg, char** args) {

	if(narg != 2) {
		std::cout << "usage: hiprmc <config_file>" << std::endl;
		return 0;
	} // if

	hir::RMC my_rmc(narg, args, args[1]);
	my_rmc.simulate_and_scale();

	return 0;
} // main()
