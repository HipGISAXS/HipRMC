/***
  *  Project: HipRMC
  *
  *  File: hiprmc.cpp
  *  Created: Jan 27, 2013
  *  Modified: Sun 27 Jan 2013 01:30:38 PM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */


#include <iostream>

#include "rmc.hpp"

int main(int narg, char** args) {

	float *loading = new (std::nothrow) float[2];
	loading[0] = 0.5;
	loading[1] = 0.25;

	hir::RMC<float> my_rmc(10, 10, "/home/asarje/hiprmc.git/data/image.tif", 2, loading);

	delete[] loading;
	return 0;
} // main()
