/***
  *  Project: HipRMC
  *
  *  File: hiprmc.cpp
  *  Created: Jan 27, 2013
  *  Modified: Thu 31 Jan 2013 03:03:05 PM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */


#include <iostream>
#include <complex>
#include <cuComplex.h>

#include "rmc.hpp"

int main(int narg, char** args) {

	float *loading = new (std::nothrow) float[2];
	loading[0] = 0.5;
	loading[1] = 0.25;

	std::string img("/home/asarje/hiprmc.git/data/image_02.tif");
	hir::RMC<float, std::complex<float>, cuFloatComplex> my_rmc(8, 8, img.c_str(), 1, loading);
	my_rmc.simulate(1000, 1, 1);

	delete[] loading;
	return 0;
} // main()
