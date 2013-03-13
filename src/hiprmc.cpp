/***
  *  Project: HipRMC
  *
  *  File: hiprmc.cpp
  *  Created: Jan 27, 2013
  *  Modified: Tue 12 Mar 2013 01:46:42 PM PDT
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */


#include <iostream>
#include <woo/timer/woo_boostchronotimers.hpp>

#include "rmc.hpp"

int main(int narg, char** args) {

	if(narg != 7) {
		std::cout << "usage: hiprmc <image_file> <size> <num steps factor> <init model size> <tile0 loading> <scaling factor>" << std::endl;
		return 0;
	} // if

	hir::real_t *loading = new (std::nothrow) hir::real_t[2];
	loading[0] = atof(args[5]); //0.0625;
	loading[1] = 0.4;
	std::string img(args[1]);
	unsigned int size = atoi(args[2]);
	unsigned int num_steps_fac = atoi(args[3]);
	unsigned int init_size = atoi(args[4]);
	unsigned int scale_factor = atoi(args[6]);

	woo::BoostChronoTimer mytimer;
	mytimer.start();
	hir::RMC my_rmc(size, size, img.c_str(), 1, init_size, loading);
	mytimer.stop();
	double init_time = mytimer.elapsed_msec();
	mytimer.start();
	my_rmc.simulate_and_scale(num_steps_fac, scale_factor, 1, 10000);
	//my_rmc.simulate(num_steps, 1, 10000);
	mytimer.stop();
	std::cout << "Initialization time: " << init_time << " ms." << std::endl;
	std::cout << "Simulation time: " << mytimer.elapsed_msec() << " ms." << std::endl;

	delete[] loading;
	return 0;
} // main()
