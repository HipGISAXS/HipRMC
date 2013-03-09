/***
  *  $Id: image.hpp 40 2012-08-21 20:43:52Z asarje $
  *
  *  Project: HipGISAXS (High-Performance GISAXS)
  *
  *  File: image.hpp
  *  Created: Jun 18, 2012
  *  Modified: Sat 09 Mar 2013 02:15:39 PM PST
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

	class Image {
		//template <typename ChannelValue, typename Layout> struct pixel;
		//typedef pixel<bits8, rgb_layout_t> rgb8_pixel_t;

		private:
			unsigned int nx_;				/* x dimension - used in 3D image construction */
			unsigned int ny_;				/* y dimension */
			unsigned int nz_;				/* z dimension */
			boost::gil::rgb8_pixel_t* image_buffer_;	/* this will hold the final rgb values */
			ColorMap8 color_map_;			/* defines mapping to colors in the defined palette */
			ColorMap new_color_map_;		/* new color mapping */

			bool convert_to_rgb_pixels(unsigned int, unsigned int, real_t*);
			bool convert_to_rgb_palette(unsigned int, unsigned int, real_t*);
			bool slice(Image* &img, unsigned int xval = 0);	/* obtain a slice at given x in case of 3D data */

			bool translate_pixels_to_positive(unsigned int nx, unsigned int ny, real_t* &data);
			bool normalize_pixels(unsigned int nx, unsigned int ny, real_t* &data);
			vector2_t minmax(unsigned int n, real_t* data);

			bool scale_image(unsigned int, unsigned int, unsigned int, unsigned int, real_t*, real_t*&);
			bool resample_pixels(unsigned int, unsigned int, real_t*, unsigned int, unsigned int, real_t*&,
									const boost::gil::matrix3x2<real_t>&);

		public:
			Image(unsigned int ny, unsigned int nz);					/* initialize a 2D image object */
			Image(unsigned int ny, unsigned int nz, char* palette);
			Image(unsigned int ny, unsigned int nz, std::string palette);
			Image(unsigned int nx, unsigned int ny, unsigned int nz);	/* initialize a 3D image object */
			Image(unsigned int nx, unsigned int ny, unsigned int nz, char* palette);
			Image(unsigned int nx, unsigned int ny, unsigned int nz, std::string palette);
			Image(unsigned int nx, unsigned int ny, unsigned int nz,
					unsigned int r, unsigned int g, unsigned int b);
			~Image();

			bool construct_image(const real_t* data, int slice);
			bool construct_image(real_t* data);
			bool construct_image_direct(real_t* data);
			bool construct_palette(real_t* data);
			bool save(std::string filename);			/* save the current image buffer */
			bool save(std::string filename, int xval);	/* save slice xval */
			bool save(char* filename, int xval);
			bool save(std::string filename, int xbegin, int xend);	/* save slices from xbegin to xend */
			bool save(char* filename, int xbegin, int xend);

	}; // class Image

} // namespace wil

#endif /* _IMAGE_HPP_ */
