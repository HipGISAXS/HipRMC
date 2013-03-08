/***
  *  Project: HipRMC
  *
  *  File: hiprmc.cpp
  *  Created: Jan 27, 2013
  *  Modified: Wed 06 Mar 2013 08:05:48 PM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */


#include <iostream>
#include <woo/timer/woo_boostchronotimers.hpp>

#include "rmc.hpp"

int main(int narg, char** args) {

	if(narg != 4) {
		std::cout << "usage: hiprmc <image_file> <size> <num steps>" << std::endl;
		return 0;
	} // if

	float *loading = new (std::nothrow) float[2];
	loading[0] = 0.0625;
	loading[1] = 0.4;
	std::string img(args[1]);
	unsigned int size = atoi(args[2]);
	unsigned int num_steps = atoi(args[3]);

	woo::BoostChronoTimer mytimer;
	mytimer.start();
	hir::RMC my_rmc(size, size, img.c_str(), 1, loading);
	mytimer.stop();
	double init_time = mytimer.elapsed_msec();
	mytimer.start();
	my_rmc.simulate(num_steps, 1, 10000);
	mytimer.stop();
	std::cout << "Initialization time: " << init_time << " ms." << std::endl;
	std::cout << "Simulation time: " << mytimer.elapsed_msec() << " ms." << std::endl;

	delete[] loading;
	return 0;
} // main()
