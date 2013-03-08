/***
  *  Project: HipRMC
  *
  *  File: hiprmc.cpp
  *  Created: Jan 27, 2013
  *  Modified: Tue 05 Mar 2013 02:52:20 PM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */


#include <iostream>
#include <woo/timer/woo_boostchronotimers.hpp>

#include "rmc.hpp"

int main(int narg, char** args) {

	float *loading = new (std::nothrow) float[2];
	loading[0] = 0.25;
	loading[1] = 0.4;
	std::string img("data/image_02_256x256.tiff");

	woo::BoostChronoTimer mytimer;
	mytimer.start();
	unsigned int input_size = atoi(args[1]);
	hir::RMC my_rmc(input_size, input_size, img.c_str(), 1, loading);
	mytimer.stop();
	double init_time = mytimer.elapsed_msec();
	mytimer.start();
	unsigned int final_size = atoi(args[2]);
	my_rmc.scale(final_size);
	mytimer.stop();

	std::cout << "Initialization time: " << init_time << " ms." << std::endl;
	std::cout << "Scaling time: " << mytimer.elapsed_msec() << " ms." << std::endl;

	delete[] loading;
	return 0;
} // main()
