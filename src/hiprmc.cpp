/***
  *  Project: HipRMC
  *
  *  File: hiprmc.cpp
  *  Created: Jan 27, 2013
  *  Modified: Sun 03 Feb 2013 12:15:02 PM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */


#include <iostream>
#include <complex>
#include <cuComplex.h>
#include <woo/timer/woo_boostchronotimers.hpp>

#include "rmc.hpp"

int main(int narg, char** args) {

	float *loading = new (std::nothrow) float[2];
	loading[0] = 0.25;
	loading[1] = 0.4;
	std::string img("/home/asarje/hiprmc.git/data/image_02_256x256.tiff");

	woo::BoostChronoTimer mytimer;
	mytimer.start();
	//hir::RMC<float, std::complex<float>, cuFloatComplex> my_rmc(256, 256, img.c_str(), 2, loading);
	hir::RMC my_rmc(256, 256, img.c_str(), 2, loading);
	mytimer.stop();
	double init_time = mytimer.elapsed_msec();
	mytimer.start();
	my_rmc.simulate(10000, 1, 1000);
	mytimer.stop();
	std::cout << "Initialization time: " << init_time << " ms." << std::endl;
	std::cout << "Simulation time: " << mytimer.elapsed_msec() << " ms." << std::endl;

	delete[] loading;
	return 0;
} // main()
