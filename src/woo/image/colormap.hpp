/***
  *  Project: WOO Image Library
  *
  *  File: colormap.hpp
  *  Created: Jul 02, 2012
  *  Modified: Sun 25 Aug 2013 09:24:07 AM PDT
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef _COLORMAP_HPP_
#define _COLORMAP_HPP_

#include <cmath>
#include <boost/array.hpp>

#include "globals.hpp"

namespace wil {		// woo image library

	typedef boost::array <unsigned char, 3> color8_t;
	typedef boost::array <unsigned int, 3> palette_t;

	const double PI_ = 3.14159265358979323846;


	/**
	 * 8-bit color map
	 */	
	class ColorMap8 {
		private:
			unsigned char* color_palette_;	// make this a vector ... ?
			int palette_num_;
			unsigned int palette_size_;
	
		public:
			ColorMap8() { color_palette_ = NULL; init("jet"); } 	// default is jet
			ColorMap8(const char* palette) { color_palette_ = NULL; init(palette); }
			ColorMap8(const std::string palette) { color_palette_ = NULL; init(palette.c_str()); }
			~ColorMap8() { if(color_palette_ != NULL) delete[] color_palette_; }

			int palette_size() const { return palette_size_; }

			color8_t operator[](unsigned int index) {
				color8_t colors;
				colors[0] = colors[1] = colors[2] = 0;
				if(index >= palette_size_) {
					std::cerr << "error: color map index is out of range: "
							<< index << " / " << palette_size_ << std::endl;
					return colors;
				} // if

				colors[0] = color_palette_[index * 3];
				colors[1] = color_palette_[index * 3 + 1];
				colors[2] = color_palette_[index * 3 + 2];

				return colors;
			} // operator[]() */
	
			bool color_map(unsigned int index,
							unsigned char& red, unsigned char& green, unsigned char& blue) {
				if(index >= palette_size_) {
					std::cerr << "error: color map index is out of range: " << index << std::endl;
					return false;
				} // if
	
				red = color_palette_[index * 3];
				green = color_palette_[index * 3 + 1];
				blue = color_palette_[index * 3 + 2];

				return true;
			} // color_map()

		private:
			bool init(const char* palette_name) {
				if(palette_name == "jet") palette_num_ = 0;
				else {
					std::cerr << "error: palette '" << palette_name << "' is not defined" << std::endl;
					return false;
				} // if-else
			
				switch(palette_num_) {
					case 1:
					case 2:
			
					case 0:		// the default case - jet color map
					default:
						palette_size_ = 72;		// 72 colors (RGB)
						unsigned char color_palette[] = {
											  0,   0, 127,
											  0,   0, 141,
											  0,   0, 155,
											  0,   0, 169,
											  0,   0, 183,
											  0,   0, 198,
											  0,   0, 212,
											  0,   0, 226,
											  0,   0, 240,
											  0,   0, 255,
											  0,  14, 255,
											  0,  28, 255,
											  0,  42, 255,
											  0,  56, 255,
											  0,  70, 255,
											  0,  84, 255,
											  0,  98, 255,
											  0, 112, 255,
											  0, 127, 255,
											  0, 141, 255,
											  0, 155, 255,
											  0, 169, 255,
											  0, 183, 255,
											  0, 198, 255,
											  0, 212, 255,
											  0, 226, 255,
											  0, 240, 255,
											  0, 255, 255,
											 14, 255, 240,
											 28, 255, 226,
											 42, 255, 212,
											 56, 255, 198,
											 70, 255, 183,
											 84, 255, 169,
											 98, 255, 155,
											112, 255, 141,
											127, 255, 127,
											141, 255, 112,
											155, 255,  98,
											169, 255,  84,
											183, 255,  70,
											198, 255,  56,
											212, 255,  42,
											226, 255,  28,
											240, 255,  14,
											255, 255,   0,
											255, 240,   0,
											255, 226,   0,
											255, 212,   0,
											255, 198,   0,
											255, 183,   0,
											255, 169,   0,
											255, 155,   0,
											255, 141,   0,
											255, 127,   0,
											255, 112,   0,
											255,  98,   0,
											255,  84,   0,
											255,  70,   0,
											255,  56,   0,
											255,  42,   0,
											255,  28,   0,
											255,  14,   0,
											255,   0,   0,
											240,   0,   0,
											226,   0,   0,
											212,   0,   0,
											198,   0,   0,
											183,   0,   0,
											169,   0,   0,
											155,   0,   0,
											141,   0,   0
										};
						color_palette_ = new (std::nothrow) unsigned char[palette_size_ * 3];
						for(unsigned int i = 0; i < palette_size_ * 3; ++ i)
							color_palette_[i] = color_palette[i];
				} // switch
			
				return true;
			} // init()
	
	}; // class ColorMap8


	/**
	 * Full color map
	 */
	class ColorMap {
		public:
			ColorMap() {
				palette_[0] = 38;		// default palette
				palette_[1] = 39;
				palette_[2] = 40;
				construct_channel_limits();
			} // ColorMap()	

			ColorMap(unsigned int red_f, unsigned green_f, unsigned int blue_f) {
				if(red_f < 0 || red_f > 40 ||
						green_f < 0 || green_f > 40 ||
						blue_f < 0 || blue_f > 40) {
					std::cerr << "error: cannot initialize ColorMap with "
								<< "specified palette parameters" << std::endl;
					exit(1);
				} // if
				palette_[0] = red_f;
				palette_[1] = green_f;
				palette_[2] = blue_f;
				construct_channel_limits();
			} // ColorMap()

			~ColorMap() { }

			void construct_channel_limits() {
				//channel_limits_[0][0] = 0.0; channel_limits_[0][1] = 0.0;
				channel_limits_[0][0] = 0.0; channel_limits_[0][1] = 1.0;
				//channel_limits_[1][0] = 0.5; channel_limits_[1][1] = 0.5;
				channel_limits_[1][0] = 0.5; channel_limits_[1][1] = 1.0;
				channel_limits_[2][0] = 1.0; channel_limits_[2][1] = 1.0;
				channel_limits_[3][0] = 0.0; channel_limits_[3][1] = 1.0;
				channel_limits_[4][0] = 0.0; channel_limits_[4][1] = 1.0;
				channel_limits_[5][0] = 0.0; channel_limits_[5][1] = 1.0;
				channel_limits_[6][0] = 0.0; channel_limits_[6][1] = 1.0;
				channel_limits_[7][0] = 0.0; channel_limits_[7][1] = 1.0;
				channel_limits_[8][0] = 0.0; channel_limits_[8][1] = 1.0;
				channel_limits_[9][0] = -1.0; channel_limits_[9][1] = 1.0;
				channel_limits_[10][0] = -1.0; channel_limits_[10][1] = 1.0;
				//channel_limits_[11][0] = 0.0; channel_limits_[11][1] = 0.5;
				channel_limits_[11][0] = 0.0; channel_limits_[11][1] = 1.0;
				channel_limits_[12][0] = 0.0; channel_limits_[12][1] = 1.0;
				channel_limits_[13][0] = -1.0; channel_limits_[13][1] = 1.0;
				channel_limits_[14][0] = 0.0; channel_limits_[14][1] = 1.0;
				channel_limits_[15][0] = -1.0; channel_limits_[15][1] = 1.0;
				channel_limits_[16][0] = -1.0; channel_limits_[16][1] = 1.0;
				channel_limits_[17][0] = 0.0; channel_limits_[17][1] = 1.0;
				channel_limits_[18][0] = 0.0; channel_limits_[18][1] = 1.0;
				channel_limits_[19][0] = 0.0; channel_limits_[19][1] = 1.0;
				channel_limits_[20][0] = 0.0; channel_limits_[20][1] = 1.0;
				channel_limits_[21][0] = 0.0; channel_limits_[21][1] = 3.0;
				channel_limits_[22][0] = -1.0; channel_limits_[22][1] = 2.0;
				channel_limits_[23][0] = -2.0; channel_limits_[23][1] = 1.0;
				channel_limits_[24][0] = 0.0; channel_limits_[24][1] = 2.0;
				channel_limits_[25][0] = 0.0; channel_limits_[25][1] = 2.0;
				channel_limits_[26][0] = -0.5; channel_limits_[26][1] = 1.0;
				//channel_limits_[27][0] = -1.0; channel_limits_[27][1] = 0.5;
				channel_limits_[27][0] = -1.0; channel_limits_[27][1] = 1.0;
				channel_limits_[28][0] = 0.0; channel_limits_[28][1] = 1.0;
				channel_limits_[29][0] = 0.0; channel_limits_[29][1] = 1.0;
				channel_limits_[30][0] = -0.78125; channel_limits_[30][1] = 2.34375;
				channel_limits_[31][0] = -0.84; channel_limits_[31][1] = 1.16;
				channel_limits_[32][0] = -0.16; channel_limits_[32][1] = 1.84;
				channel_limits_[33][0] = 0.0; channel_limits_[33][1] = 1.5;
				channel_limits_[34][0] = 0.0; channel_limits_[34][1] = 2.0;
				channel_limits_[35][0] = -0.5; channel_limits_[35][1] = 1.5;
				channel_limits_[36][0] = -1.0; channel_limits_[36][1] = 1.0;
				channel_limits_[37][0] = -11.5; channel_limits_[37][1] = 1.0;
				channel_limits_[38][0] = 0.0; channel_limits_[38][1] = 1.0;
				channel_limits_[39][0] = 0.0; channel_limits_[39][1] = 1.0;
				channel_limits_[40][0] = 0.0; channel_limits_[40][1] = 1.0;
			} // construct_channel_limits()

			color8_t color_map(double value) {
				if(value < 0.0 || value > 1.0) {
					std::cerr << "error: color value is outside [0, 1]" << std::endl;
					exit(1);
				} // if
				color8_t channels;
				channels[0] = channel_map(0, value);
				channels[1] = channel_map(1, value);
				channels[2] = channel_map(2, value);
				return channels;
			} // color_map()

		private:
			palette_t palette_;
			double channel_limits_[41][2];

			unsigned int channel_map(unsigned int channel, double value) {
				unsigned int func_num = palette_[channel];
				double channel_val = compute_channel(func_num, value);	// channel_val is >= 0 and <= 1
				double channel_max = compute_channel(func_num, 1.0);	// make a lookup tabke for this
				double channel_min = compute_channel(func_num, 0.0);
				unsigned int result = std::floor(((channel_val - channel_limits_[func_num][0]) /
								(channel_limits_[func_num][1] - channel_limits_[func_num][0])) * 255.0);
				if(result > 255) std::cerr << "error: color value more than 255: " << result << std::endl;
				return result;
			} // channel_map()

			double compute_channel(unsigned int func_num, double x) {
				double temp;
				switch(func_num) {
					case 0:
						return 0.0;

					case 1:
						return 0.5;

					case 2:
						return 1.0;

					case 3:
						return x;

					case 4:
						return x * x;

					case 5:
						return pow(x, 3);

					case 6:
						return pow(x, 4);

					case 7:
						return sqrt(x);

					case 8:
						return sqrt(sqrt(x));

					case 9:
						temp = sin(hir::PI_ * x);
						return temp;// < 0.0 ? 0.0 : temp;

					case 10:
						temp = cos(hir::PI_ * x);
						return temp;// < 0.0 ? 0.0 : temp;

					case 11:
						return fabs(x - 0.5);

					case 12:
						return pow((2 * x - 1.0), 2);

					case 13:
						temp = sin(2 * hir::PI_ * x);
						return temp;// < 0.0 ? 0.0 : temp;

					case 14:
						return fabs(cos(2 * hir::PI_ * x));

					case 15:
						temp = sin(4 * hir::PI_ * x);
						return temp;// < 0.0 ? 0.0 : temp;

					case 16:
						temp = cos(4 * hir::PI_ * x);
						return temp;// < 0.0 ? 0.0 : temp;

					case 17:
						return fabs(sin(4.0 * hir::PI_ * x));

					case 18:
						return fabs(cos(4.0 * hir::PI_ * x));

					case 19:
						return fabs(sin(8.0 * hir::PI_ * x));

					case 20:
						return fabs(cos(8.0 * hir::PI_ * x));

					case 21:
						return 3.0 * x;

					case 22:
						temp = 3.0 * x - 1.0;
						return temp;// < 0.0 ? 0.0 : temp;

					case 23:
						temp = 3.0 * x - 2.0;
						return temp;// < 0.0 ? 0.0 : temp;

					case 24:
						return fabs(3.0 * x - 1.0);

					case 25:
						return fabs(3.0 * x - 2.0);

					case 26:
						temp = (3.0 * x - 1.0) / 2.0;
						return temp;// < 0.0 ? 0.0 : temp;

					case 27:
						temp = (3.0 * x - 2.0) / 2.0;
						return temp;// < 0.0 ? 0.0 : temp;

					case 28:
						return fabs((3.0 * x - 1.0) / 2.0);

					case 29:
						return fabs((3.0 * x - 2.0) / 2.0);

					case 30:
						temp = x / 0.32 - 0.78125;
						return temp;// < 0.0 ? 0.0 : temp;

					case 31:
						temp = 2 * x - 0.84;
						return temp;// < 0.0 ? 0.0 : temp;

					case 32:
						temp = 1.84 - 2.0 * x;
						return temp;// < 0.0 ? 0.0 : temp;

					case 33:
						return fabs(2.0 * x - 0.5);

					case 34:
						return 2.0 * x;

					case 35:
						temp = 2.0 * x - 0.5;
						return temp;// < 0.0 ? 0.0 : temp;

					case 36:
						temp = 2.0 * x - 1.0;
						return temp;// < 0.0 ? 0.0 : temp;

					case 37:					// another one for case 32
						temp = x / 0.08 - 11.5;
						return temp;// < 0.0 ? 0.0 : temp;

					case 38:
						//temp = cos(PI_ / 2 * (x - 1));
						//temp = sin(4 * PI_ * x / 3 - 9 * PI_ / 16);
						temp = sin(25 * hir::PI_ * x / 24 - 7 * hir::PI_ / 32);
						return temp < 0.0 ? 0.0 : temp;

					case 39:
						//temp = - sin(7 * PI_ / 6 * (x - 1));
						//temp = -sin(3 * PI_ * x / 2 + 5 * PI_ / 8);
						temp = -sin(4 * hir::PI_ * x / 3 + 13 * hir::PI_ / 16);
						return temp < 0.0 ? 0.0 : temp;

					case 40:
						//temp = sin(2 * PI_ * x);
						//temp = -sin(7 * PI_ * x / 4 - 7 * PI_ / 8);
						temp = -sin(4 * hir::PI_ * x / 3 - 7 * hir::PI_ / 8);
						return temp < 0.0 ? 0.0 : temp;

					default:
						std::cerr << "error: color function out of bounds" << std::endl;
						exit(1);

						/*	0: 0               1: 0.5             2: 1
							3: x               4: x^2             5: x^3
							6: x^4             7: sqrt(x)         8: sqrt(sqrt(x))
							9: sin(90x)        10: cos(90x)       11: |x-0.5|
							12: (2x-1)^2       13: sin(180x)      14: |cos(180x)|
							15: sin(360x)      16: cos(360x)      17: |sin(360x)|
							18: |cos(360x)|    19: |sin(720x)|    20: |cos(720x)|
							21: 3x             22: 3x-1           23: 3x-2
							24: |3x-1|         25: |3x-2|         26: (3x-1)/2
							27: (3x-2)/2       28: |(3x-1)/2|     29: |(3x-2)/2|
							30: x/0.32-0.78125 31: 2*x-0.84       32: 4x;1;-2x+1.84;x/0.08-11.5
							33: |2*x - 0.5|    34: 2*x            35: 2*x - 0.5
							36: 2*x - 1 */
				} // switch
			} // compute_channel()

	}; // class ColorMap

} // namespace wil
	
#endif /* _COLORMAP_HPP_ */
