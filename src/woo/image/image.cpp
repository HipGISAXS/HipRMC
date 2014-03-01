/***
  *  Project: WOO Image Library
  *
  *  File: image.cpp
  *  Created: Jun 18, 2012
  *  Modified: Sun 25 Aug 2013 09:24:10 AM PDT
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#include <boost/math/special_functions/round.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/gil/extension/io/tiff_io.hpp>
#include <boost/gil/extension/numeric/sampler.hpp>
#include <boost/gil/extension/numeric/resample.hpp>

#include "image.hpp"
#include "utilities.hpp"

namespace wil {

	Image::Image(unsigned int ny, unsigned int nz):
					nx_(1), ny_(ny), nz_(nz), color_map_8_() {
		image_buffer_ = NULL;
	} // Image::Image()


	Image::Image(unsigned int ny, unsigned int nz, char* palette):
					nx_(1), ny_(ny), nz_(nz), color_map_8_(palette) {
		image_buffer_ = NULL;
	} // Image::Image()


	Image::Image(unsigned int ny, unsigned int nz, std::string palette):
					nx_(1), ny_(ny), nz_(nz), color_map_8_(palette) {
		image_buffer_ = NULL;
	} // Image::Image()


	Image::Image(unsigned int ny, unsigned int nz, unsigned int r, unsigned int g, unsigned int b):
					nx_(1), ny_(ny), nz_(nz), color_map_(r, g, b) {
		image_buffer_ = NULL;
	} // Image::Image()


	Image::Image(unsigned int nx, unsigned int ny, unsigned int nz):
					nx_(nx), ny_(ny), nz_(nz), color_map_8_(), color_map_() {
		image_buffer_ = NULL;
	} // Image::Image()


	Image::Image(unsigned int nx, unsigned int ny, unsigned int nz, char* palette):
					nx_(nx), ny_(ny), nz_(nz), color_map_8_(palette) {
		image_buffer_ = NULL;
	} // Image::Image()


	Image::Image(unsigned int nx, unsigned int ny, unsigned int nz, std::string palette):
					nx_(nx), ny_(ny), nz_(nz), color_map_8_(palette) {
		image_buffer_ = NULL;
	} // Image::Image()


	Image::Image(unsigned int nx, unsigned int ny, unsigned int nz,
			unsigned int r, unsigned int g, unsigned int b):
					nx_(nx), ny_(ny), nz_(nz), color_map_(r, g, b) {
		image_buffer_ = NULL;
	} // Image::Image()


	Image::~Image() {
		if(image_buffer_ != NULL) delete[] image_buffer_;
		image_buffer_ = NULL;
	} // Image::~Image()


	void print_arr_2d(real_t* data, unsigned int nx, unsigned int ny) {
		for(unsigned int i = 0; i < ny; ++ i) {
			for(unsigned int j = 0; j < nx; ++ j) {
				std::cout << data[nx * i + j] << "\t";
			} // for
			std::cout << std::endl;
		} // for
	} // print_arr_2d()


	void print_rgb_2d(boost::gil::rgb8_pixel_t* data, unsigned int nx, unsigned int ny) {
		for(unsigned int i = 0; i < ny; ++ i) {
			for(unsigned int j = 0; j < nx; ++ j) {
				std::cout << data[nx * i + j][0] << "," << data[nx * i + j][1]
							<< "," << data[nx * i + j][2] << "\t";
			} // for
			std::cout << std::endl;
		} // for
	} // print_arr_2d()


	/**
	 * an overload of construct_image to first create a slice from data
	 * and then contruct the image for that slice
	 */
	bool Image::construct_image(const real_t* data_3d, int xslice) {		// improve the structure here ...
		if(ny_ < 1 || nz_ < 1) return false;

		real_t* slice_data = new (std::nothrow) real_t[ny_ * nz_];
		for(unsigned int i = 0; i < nz_; ++ i) {
			for(unsigned int j = 0; j < ny_; ++ j) {
				slice_data[ny_ * i + j] = data_3d[nx_ * ny_ * i + nx_ * j + xslice];
			} // for
		} // for
		nx_ = 1; 		// not a good fix ... do something better
		bool ret = construct_image(slice_data);
		delete[] slice_data;
		return ret;
	} // Image::construct_image()


	/**
	 * given a 2d/3d array of real values, construct an image
	 * in case of 3d (not implemented), nx_ images will be created into image_buffer_
	 */
	// parallelize this for multicore
	bool Image::construct_log_image(real_t* data) {						// and here ...
		if(data == NULL) {
			std::cerr << "empty data found while constructing image" << std::endl;
			return false;
		} // if
		if(nx_ == 1) {	// a single slice
			if(!translate_pixels_to_positive(ny_, nz_, data)) {
				std::cerr << "error: something went awfully wrong in data translation" << std::endl;
				return false;
			} // if

			// apply log(10) on the resulting data
			if(!mat_log10_2d(ny_, nz_, data)) {
				std::cerr << "error: something went wrong in mat_log10_2d" << std::endl;
				return false;
			} // if

			if(!normalize_pixels(ny_, nz_, data)) {
				std::cerr << "error: something went awfully wrong in pixel normalization" << std::endl;
				return false;
			} // if

			// construct image_buffer_ with rgb values for each point in data
			if(!convert_to_rgb_pixels(ny_, nz_, data)) {
				std::cerr << "error: something went terribly wrong in convert_to_rgb_pixels" << std::endl;
				return false;
			} // if
		} else {
			std::cerr << "uh-oh: the case of constructing 3D image "
						<< "has not been implemented yet" << std::endl;
			return false;
			for(unsigned int x = 0; x < nx_; ++ x) {
				// save in a way so that taking slices becomes easier later
				// ...
			} // for
		} // if-else

		return true;
	} // Image::construct_log_image()


	bool Image::construct_image(real_t* data) {						// and here ...
		if(data == NULL) {
			std::cerr << "empty data found while constructing image" << std::endl;
			return false;
		} // if
		if(nx_ == 1) {	// a single slice
			//if(!translate_pixels_to_positive(ny_, nz_, data)) {
			//	std::cerr << "error: something went awfully wrong in data translation" << std::endl;
			//	return false;
			//} // if

			if(!normalize_pixels(ny_, nz_, data)) {
				std::cerr << "error: something went awfully wrong in pixel normalization" << std::endl;
				return false;
			} // if

			// construct image_buffer_ with rgb values for each point in data
			if(!convert_to_rgb_pixels(ny_, nz_, data)) {
				std::cerr << "error: something went terribly wrong in convert_to_rgb_pixels" << std::endl;
				return false;
			} // if
		} else {
			std::cerr << "uh-oh: the case of constructing 3D image "
						<< "has not been implemented yet" << std::endl;
			return false;
			for(unsigned int x = 0; x < nx_; ++ x) {
				// save in a way so that taking slices becomes easier later
				// ...
			} // for
		} // if-else

		return true;
	} // Image::construct_image()


	bool Image::construct_palette(real_t* data) {						// and here ...
		if(data == NULL) {
			std::cerr << "empty data found while constructing image" << std::endl;
			return false;
		} // if
		if(nx_ == 1) {	// a single slice
			std::cout << "  -- Mapping data to the color palette ..." << std::endl;
			if(!convert_to_rgb_palette(ny_, nz_, data)) {
				std::cerr << "error: something went terribly wrong in convert_to_rgb_palette" << std::endl;
				return false;
			} // if
		} else {
			std::cerr << "uh-oh: the case of constructing 3D palette "
						<< "has not been implemented yet" << std::endl;
			return false;
		} // if-else
		return true;
	} // Image::construct_palette()


	vector2_t Image::minmax(unsigned int n, real_t* data) {
		vector2_t val(data[0], data[0]);
		for(unsigned int i = 0; i < n; ++ i) {
			val[0] = (val[0] > data[i]) ? data[i] : val[0];
			val[1] = (val[1] < data[i]) ? data[i] : val[1];
		} // for
		return val;
	} // Image::minmax()


	bool Image::translate_pixels_to_positive(unsigned int nx, unsigned int ny, real_t* &pixels) {
		vector2_t pixel_minmax = minmax(nx * ny, pixels);
		if(pixel_minmax[0] < 0) {
			for(unsigned int i = 0; i < nx * ny; ++ i) {
				pixels[i] = pixels[i] - pixel_minmax[0];
				if(pixels[i] < 0)
					std::cout << "oho oho ... its less than 0 ... something is gravely wrong: "
								<< pixels[i] << " (" << pixel_minmax[0] << ", " << pixel_minmax[1] << ")"
								<< std::endl;
			} // for
		} //  if
		return true;
	} // Image::translate_pixels()


	/**
	 * Normalize data in pixels to be in [0, 1]
	 */
	bool Image::normalize_pixels(unsigned int nx, unsigned int ny, real_t* &pixels) {
		vector2_t pixel_minmax = minmax(nx * ny, pixels);
		if(pixel_minmax[0] == pixel_minmax[1]) {	// all pixels have the same value
			if(pixel_minmax[0] < 0) for(unsigned int i = 0; i < nx * ny; ++ i) pixels[i] = 0;
			else for(unsigned int i = 0; i < nx * ny; ++ i) pixels[i] = 1;
		} else {
			for(unsigned int i = 0; i < nx * ny; ++ i) {
				//pixels[i] = (pixels[i] - pixel_minmax[0]) / (pixel_minmax[1] - pixel_minmax[0]);
				pixels[i] = fabs((pixels[i] - pixel_minmax[0]) / (pixel_minmax[1] - pixel_minmax[0]));
				if(pixels[i] < 0 || pixels[i] > 1)
					std::cerr << "oh oh ... its less than 0 ... something is gravely wrong: "
								<< pixels[i] << " (" << pixel_minmax[0] << ", " << pixel_minmax[1] << ")"
								<< std::endl;
			} // for
		} // if-else

		return true;
	} // Image::normalize_pixels()


	/**
	 * convert the real_t values in image data to mapped color
	 */
	bool Image::convert_to_rgb_pixels(unsigned int ny, unsigned int nz, real_t* image) {
		// assuming: values in image are in [0, 1]
		if(image_buffer_ != NULL) { delete[] image_buffer_; image_buffer_ = NULL; }
		image_buffer_ = new (std::nothrow) boost::gil::rgb8_pixel_t[ny * nz];
		if(image_buffer_ == NULL) {
			std::cerr << "error: could not allocate memory for image buffer. size = "
					<< ny << "x" << nz << std::endl;
			return false;
		} // if
		for(unsigned int i = 0; i < ny * nz; ++ i) {	// assuming 0 <= image[i] <= 1
			if(image[i] < 0 || image[i] > 1.0) {
				std::cerr << "a pixel value not within range: " << image[i] << std::endl;
				return false;
			} // if
			boost::array<unsigned char, 3> color_rgb = color_map_.color_map(image[i]);
			boost::gil::rgb8_pixel_t temp =
							boost::gil::rgb8_pixel_t(color_rgb[0], color_rgb[1], color_rgb[2]);
			image_buffer_[i] = temp;
		} // for

		return true;
	} // Image::convert_to_rgb_pixels()


	bool Image::convert_to_rgb_palette(unsigned int ny, unsigned int nz, real_t* image) {
		// assuming: values in image are in [0, 1]
		if(image_buffer_ != NULL) { delete[] image_buffer_; image_buffer_ = NULL; }
		image_buffer_ = new (std::nothrow) boost::gil::rgb8_pixel_t[ny * nz];
		if(image_buffer_ == NULL) {
			std::cerr << "error: could not allocate memory for image buffer. size = "
					<< ny << "x" << nz << std::endl;
			return false;
		} // if
		for(unsigned int i = 0; i < ny * nz; ++ i) {	// assuming 0 <= image[i] <= 1
			if(image[i] < 0 || image[i] > 1.0) {
				std::cerr << "a pixel value not within range: " << image[i] << std::endl;
				return false;
			} // if
			boost::array<unsigned char, 3> color_rgb = color_map_.color_map(image[i]);
			boost::gil::rgb8_pixel_t temp =
							boost::gil::rgb8_pixel_t(color_rgb[0], color_rgb[1], color_rgb[2]);
			image_buffer_[i] = temp;
		} // for

		return true;
	} // Image::convert_to_rgb_pixels()


	/**
	 * construct a 2D image slice from existing 3D image object
	 */
	bool Image::slice(Image* &img, unsigned int xval) {
		if(xval >= nx_) {
			std::cerr << "error: the requested slice does not exist" << std::endl;
			return false;
		} // Image::slice()

		img = new Image(ny_, nz_);		// where to delete this? ...

		for(unsigned int y = 0; y < ny_; y ++) {
			for(unsigned int z = 0; z < nz_; z ++) {
				img->image_buffer_[z * ny_ + y] = image_buffer_[z * ny_ * nx_ + y * nx_ + xval];
			} // for z
		} // for y

		return true;
	} // Image::slice()


	/**
	 * save image(s) to file(s)
	 */
	bool Image::save(std::string filename) {
		typedef boost::gil::type_from_x_iterator <boost::gil::rgb8_ptr_t> pixel_itr_t;
		pixel_itr_t::view_t view =
					interleaved_view(ny_, nz_, image_buffer_, ny_ * sizeof(boost::gil::rgb8_pixel_t));
		boost::gil::tiff_write_view(filename.c_str(), view);
		return true;
	} // Image::save()


	/**
	 * save slice image xval to file
	 */
	bool Image::save(std::string filename, int xval) {
		return save(filename.c_str());
	} // Image::save()


	/**
	 * scale the image data from old dimensions to the new dimensions
	 * NOTE: this requires the boost gil numeric library (it is not an official part of boost)
	 * let the coordinates of old and new be from min (0, 0) to max (old_x-1, old_y-1) (new_x-1, new_y-1)
	 */
	bool scale_image(int old_x, int old_y, int new_x, int new_y, real_t *old_data, real_t* &new_data) {
		
		typedef boost::gil::matrix3x2<real_t> matrix3x2;

		matrix3x2 temp1 = matrix3x2::get_translate(-new_x / 2.0, -new_y / 2.0);
		matrix3x2 temp2	= matrix3x2::get_scale((float_t)old_x / new_x, (float_t)old_y / new_y);
		matrix3x2 temp3 = matrix3x2::get_rotate(0);
		matrix3x2 temp4 = matrix3x2::get_translate(old_x / 2.0, old_y / 2.0);
		matrix3x2 mat = temp1 * temp2 * temp3 * temp4;
		new_data = new (std::nothrow) real_t[new_x * new_y];
		resample_pixels(old_x, old_y, old_data, new_x, new_y, new_data, mat);

		return true;
	} // Image::scale_image()


	/**
	 * nearest neighbor based sampling is used
	 */
	bool resample_pixels(int old_x, int old_y, real_t* old_data, int new_x, int new_y, real_t* &new_data,
								const boost::gil::matrix3x2<real_t>& mat) {
		// mapping from new to old
		boost::gil::point2 <int> new_p;
		for(new_p.y = 0; new_p.y < (int) new_y; ++ new_p.y) {
			for(new_p.x = 0; new_p.x < (int) new_x; ++ new_p.x) {
				boost::gil::point2 <real_t> trans_p = boost::gil::transform(mat, new_p); // this
															// is basically matrix multiplication! TODO ...
				boost::gil::point2 <int> center(boost::math::iround(trans_p.x),
												boost::math::iround(trans_p.y));
				if(center.x >= 0 && center.y >= 0 && center.x < (int) old_x &&
						center.y < (int) old_y)
					new_data[new_x * new_p.y + new_p.x] = old_data[old_x * center.y + center.x];
			} // for x
		} // for y
		return true;
	} // Image::resample_pixels()

} // namespace wil
