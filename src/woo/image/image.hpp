/***
  *  Project: WOO Image Library
  *
  *  File: image.hpp
  *  Created: Jun 18, 2012
  *  Modified: Sun 25 Aug 2013 09:24:12 AM PDT
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef _IMAGE_HPP_
#define _IMAGE_HPP_

#include <boost/gil/gil_all.hpp>
#include <boost/gil/extension/numeric/affine.hpp>

#include "globals.hpp"
#include "colormap.hpp"
#include "typedefs.hpp"

namespace wil {

	/**
	 * The main image class
	 */
	class Image {
		//template <typename ChannelValue, typename Layout> struct pixel;
		//typedef pixel<bits8, rgb_layout_t> rgb8_pixel_t;

		private:
			unsigned int nx_;				/* x dimension - used in 3D image construction */
			unsigned int ny_;				/* y dimension */
			unsigned int nz_;				/* z dimension */
			boost::gil::rgb8_pixel_t* image_buffer_;	/* this will hold the final rgb values */
			ColorMap8 color_map_8_;			/* defines mapping to colors in the defined palette */
			ColorMap color_map_;			/* better color mapping */

			bool convert_to_rgb_pixels(unsigned int, unsigned int, real_t*);
			bool convert_to_rgb_palette(unsigned int, unsigned int, real_t*);
			bool slice(Image* &img, unsigned int xval = 0);	/* obtain a slice at given x in case of 3D data */

			bool translate_pixels_to_positive(unsigned int nx, unsigned int ny, real_t* &data);
			bool normalize_pixels(unsigned int nx, unsigned int ny, real_t* &data);
			vector2_t minmax(unsigned int n, real_t* data);

		public:
			Image(unsigned int ny, unsigned int nz);					/* initialize a 2D image object */
			Image(unsigned int ny, unsigned int nz, char* palette);
			Image(unsigned int ny, unsigned int nz, std::string palette);
			Image(unsigned int ny, unsigned int nz, unsigned int r, unsigned int g, unsigned int b);
			Image(unsigned int nx, unsigned int ny, unsigned int nz);	/* initialize a 3D image object */
			Image(unsigned int nx, unsigned int ny, unsigned int nz, char* palette);
			Image(unsigned int nx, unsigned int ny, unsigned int nz, std::string palette);
			Image(unsigned int nx, unsigned int ny, unsigned int nz,
					unsigned int r, unsigned int g, unsigned int b);
			~Image();

			bool construct_image(const real_t* data, int slice);
			bool construct_log_image(real_t* data);
			bool construct_image(real_t* data);
			bool construct_palette(real_t* data);
			bool save(std::string filename);			/* save the current image buffer */
			bool save(std::string filename, int xval);	/* save slice xval */
			bool save(char* filename, int xval);
			bool save(std::string filename, int xbegin, int xend);	/* save slices from xbegin to xend */
			bool save(char* filename, int xbegin, int xend);

	}; // class Image

	bool scale_image(int, int, int, int, real_t*, real_t*&);
	bool resample_pixels(int, int, real_t*, int, int, real_t*&, const boost::gil::matrix3x2<real_t>&);

} // namespace wil

#endif /* _IMAGE_HPP_ */
