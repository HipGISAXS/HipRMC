/***
  *  Project: HipRMC - High Performance Reverse Monte Carlo Simulations
  *
  *  File: rmc.cpp
  *  Created: Jan 25, 2013
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#include <random>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <ctime>

#include "rmc.hpp"
#include "constants.hpp"
#include "hiprmc_input.hpp"
#ifdef USE_GPU
#include "init_gpu.cuh"
#endif
#ifdef _OPENMP
#include <omp.h>
#endif


namespace hir {

//#include "temp.hpp"

	RMC::RMC(int narg, char** args, char* filename) :
			#ifdef USE_MPI
				multi_node_(narg, args),
			#endif
			in_pattern_(0, 0),
			//scaled_pattern_(0, 0),
			cropped_pattern_(0, 0),
			mask_mat_(0, 0),
			cropped_mask_mat_(0, 0),
			vandermonde_mat_(0, 0) {
		if(!HipRMCInput::instance().construct_input_config(filename)) {
			std::cerr << "error: failed to construct input configuration" << std::endl;
			exit(1);
		} // if

		rows_ = HipRMCInput::instance().num_rows();
		cols_ = HipRMCInput::instance().num_cols();
		size_ = std::max(rows_, cols_);
		global_num_tiles_ = HipRMCInput::instance().num_tiles();
		num_tiles_ = global_num_tiles_;
		unsigned int start_num_rows = HipRMCInput::instance().model_start_num_rows();
		unsigned int start_num_cols = HipRMCInput::instance().model_start_num_cols();

    // if there is an initial model, don't do scaling
    if(HipRMCInput::instance().init_model().empty())
      tile_size_ = std::max(start_num_rows, start_num_cols);
    else
      tile_size_ = std::max(rows_, cols_);

		//scaled_pattern_.resize(tile_size_, tile_size_);
		//in_pattern_.resize(rows_, cols_);
		//cropped_pattern_.resize(tile_size_, tile_size_);
		//mask_mat_.resize(rows_, cols_);
		//cropped_mask_mat_.resize(tile_size_, tile_size_);
		//vandermonde_mat_.resize(tile_size_, tile_size_);

		// for now only square patterns are considered
		if(rows_ != cols_) {
			std::cerr << "error: number of rows should equal number of columns" << std::endl;
			exit(1);
		} // if
		if(tile_size_ > rows_) {
			std::cerr << "error: initial tile size should be less or equal to pattern size" << std::endl;
			exit(1);
		} // if
		#ifdef USE_MPI
			if(!multi_node_.init()) {
				std::cerr << "error: " << std::endl;
				exit(1);
			} // if
		#endif

		#ifdef USE_MPI
			if(multi_node_.is_master()) {
		#endif
				std::cout << "*****************************************************************"
							<< std::endl
							<< "***                      HipRMC v0.2 beta                     ***"
							<< std::endl
							<< "*****************************************************************"
							<< std::endl
							<< std::endl;
				HipRMCInput::instance().print_all();
		#ifdef USE_MPI
			} // if
		#endif

		#ifdef USE_GPU
			if(!init_gpu()) {
				std::cerr << "error: " << std::endl;
				exit(1);
			} // if
		#endif
		if(!init()) {
			std::cerr << "error: failed to pre-initialize RMC object" << std::endl;
			exit(1);
		} // if
	} // RMC::RMC()


	RMC::~RMC() {

	} // RMC::~RMC()


	bool RMC::init() {
		#ifdef USE_MPI
		if(multi_node_.is_master()) {
		#endif
			std::cout << "++ Initializing HipRMC ..." << std::endl;
			// create output directory
			const std::string p = HipRMCInput::instance().label();
			if(!boost::filesystem::create_directory(p)) {
				std::cerr << "error: could not create output directory " << p << std::endl;
				return false;
			} // if
		#ifdef USE_MPI
		} // if
		#endif
		#ifdef USE_MPI
  		if(multi_node_.size() == 1) {	// all tiles to one proc
	  		multi_node_.dup("real_world", "world");
		  	num_tiles_ = global_num_tiles_;
  		} else if(multi_node_.size() == global_num_tiles_) {	// each proc has exactly 1 tile
	  		multi_node_.split("real_world", "world", multi_node_.rank());
		  	num_tiles_ = 1;
  		} else if(global_num_tiles_ < multi_node_.size()) {
        // TODO:
        std::cout << "error: this case of ntiles < nodes is not completely correct" << std::endl;
        return false;
	  		// multiple processors are responsible for each tile
		  	// tile number this processor is responsible for (round robin distribution)
  			int tile_num = multi_node_.rank("world") % global_num_tiles_;
	  		// create new communicator
		  	multi_node_.split("real_world", "world", tile_num);
			  num_tiles_ = 1;
  		} else if(multi_node_.size() < global_num_tiles_) {	// each proc has multiple tiles
	  		// multiple tiles are assigned to each processor
		  	// each proc is in its own world
			  multi_node_.split("real_world", "world", multi_node_.rank());
  			num_tiles_ = (global_num_tiles_ / multi_node_.size("world")) +
	                   (multi_node_.rank("world") < global_num_tiles_ % multi_node_.size("world"));
  		} // if-else

	  	// construct a communicator in world for all masters in real_world
		  int color = multi_node_.is_master("real_world");
  		multi_node_.split("real_world_masters", "world", color);

      if(multi_node_.is_master("real_world")) {
	  		std::cout << "++     Number of MPI processes used: " << multi_node_.size("real_world")
		      				<< std::endl;
  			#pragma omp parallel
	  		{
		  		#ifdef _OPENMP
			  		if(omp_get_thread_num() == 0)
				  		std::cout << "++         Number of OpenMP threads: " << omp_get_num_threads()
					  				<< std::endl;
  				#endif
	  		} // pragma omp parallel
		  } // if
		#endif // USE_MPI

		real_t *img_data = new (std::nothrow) real_t[rows_ * cols_];
		unsigned int *mask_data = new (std::nothrow) unsigned int[rows_ * cols_];

		#ifdef USE_MPI
		if(multi_node_.is_master("world")) {
		#endif
			double min_val, max_val;
			// TODO: take a subimage of the input ...

			//////////////////////////////////////////
			wil::Image img(0, 0, 30, 30, 30);
			//if(!img.read(HipRMCInput::instance().input_image(), rows_, cols_)) return false;
			if(!img.read_tiff(HipRMCInput::instance().input_image(), rows_, cols_)) return false;
			img.get_data(img_data);
			wil::Image img_temp(rows_, cols_, 30, 30, 30);
			img_temp.construct_image(img_data);
			img_temp.save(HipRMCInput::instance().label() + "/input_image.tif");
			////////////////////////////////////////

			// read in mask if given
			if(HipRMCInput::instance().mask_set()) {
				// read mask file and set the mask matrix
				wil::Image mask_img(0, 0, 30, 30, 30);
				//if(!mask_img.read(HipRMCInput::instance().mask_image(), rows_, cols_)) return false;
				//if(!mask_img.read_new(HipRMCInput::instance().mask_image(), rows_, cols_)) return false;
				if(!mask_img.read_tiff(HipRMCInput::instance().mask_image(), rows_, cols_)) return false;
				real_t *temp_mask = new (std::nothrow) real_t[rows_ * cols_];
				mask_img.get_data(temp_mask);
				//std::cout << rows_ << " x " << cols_ << std::endl;
				for(unsigned int i = 0; i < rows_; ++ i) {
					for(unsigned int j = 0; j < cols_; ++ j) {
						unsigned int index = cols_ * i + j;
						real_t temp = temp_mask[index];
						mask_data[index] = (255 * temp < 180) ? 0 : 1;
						//std::cout << mask_data[index] << " ";
					} // for
					//std::cout << std::endl;
				} // for
				delete[] temp_mask;
			} else {	// no mask is set
				// set all entries in mask matrix to 1
				for(unsigned int i = 0; i < rows_; ++ i) {
					for(unsigned int j = 0; j < cols_; ++ j) {
						unsigned int index = cols_ * i + j;
						mask_data[index] = 1;
					} // for
				} // for
			} // if-else
			// write the mask to output (for verification)
			wil::Image mask_temp(rows_, cols_, 30, 30, 30);
			mask_temp.construct_image(mask_data);
			mask_temp.save(HipRMCInput::instance().label() + "/input_mask.tif");

			#ifdef USE_MODEL_INPUT	// for testing/debugging
			for(int i = 0; i < rows_ * cols_; ++ i) img_data[i] = (255 * img_data[i] < 180) ? 0 : 1;
			wil::Image base_01pat(rows_, cols_, 30, 30, 30);
			base_01pat.construct_image(img_data);
			base_01pat.save(HipRMCInput::instance().label() + "/base_01_pattern.tif");

			max_val = 0.0, min_val = 1e10;
			#ifdef USE_GPU
				cucomplex_t *mat_in_h, *mat_out_h;
				cucomplex_t *mat_in_d, *mat_out_d;
				mat_in_h = new (std::nothrow) cucomplex_t[rows_ * cols_];
				mat_out_h = new (std::nothrow) cucomplex_t[rows_ * cols_];
				cudaMalloc((void**) &mat_in_d, rows_ * cols_ * sizeof(cucomplex_t));
				cudaMalloc((void**) &mat_out_d, rows_ * cols_ * sizeof(cucomplex_t));
	        	for(int i = 0; i < rows_; ++ i) {
    	        	for(int j = 0; j < cols_; ++ j) {
        	        	mat_in_h[cols_ * i + j].x = (double) img_data[cols_ * i + j];
	        	        mat_in_h[cols_ * i + j].y = 0.0;
    	        	    mat_out_h[cols_ * i + j].x = 0.0;
        	        	mat_out_h[cols_ * i + j].y = 0.0;
	            	} // for
		        } // for
				cudaMemcpy(mat_in_d, mat_in_h, rows_ * cols_ * sizeof(cucomplex_t),
							cudaMemcpyHostToDevice);
				cudaMemcpy(mat_out_d, mat_out_h, rows_ * cols_ * sizeof(cucomplex_t),
							cudaMemcpyHostToDevice);
				cufftHandle plan;
				cufftPlan2d(&plan, rows_, cols_, CUFFT_Z2Z);
				cufftExecZ2Z(plan, mat_in_d, mat_out_d, CUFFT_FORWARD);
				cudaThreadSynchronize();
				cudaMemcpy(mat_out_h, mat_out_d, rows_ * cols_ * sizeof(cucomplex_t),
							cudaMemcpyDeviceToHost);
				cudaThreadSynchronize();
        		for(int i = 0; i < rows_; ++ i) {
            		for(int j = 0; j < cols_; ++ j) {
    	        	    real_t temp = pow(mat_out_h[cols_ * i + j].x, 2) +
										pow(mat_out_h[cols_ * i + j].y, 2);
						min_val = (temp < min_val) ? temp : min_val;
						max_val = (temp > max_val) ? temp : max_val;
						int i_swap = (i + (rows_ >> 1)) % rows_;
						int j_swap = (j + (cols_ >> 1)) % cols_;
						img_data[i_swap * cols_ + j_swap] = temp;
        	    	} // for
	        	} // for
				cufftDestroy(plan);
				cudaFree(mat_out_d);
				cudaFree(mat_in_d);
				delete[] mat_out_h;
				delete[] mat_in_h;
			#else	// USE CPU
				fftw_complex* mat_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * rows_ * cols_);
    	    	fftw_complex* mat_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * rows_ * cols_);
	        	for(int i = 0; i < rows_; ++ i) {
    	        	for(int j = 0; j < cols_; ++ j) {
        	        	mat_in[cols_ * i + j][0] = (double) img_data[cols_ * i + j];
	        	        mat_in[cols_ * i + j][1] = 0.0;
    	        	    mat_out[cols_ * i + j][0] = 0.0;
        	        	mat_out[cols_ * i + j][1] = 0.0;
	            	} // for
		        } // for
				fftw_plan plan;
        		plan = fftw_plan_dft_2d(rows_, cols_, mat_in, mat_out, FFTW_FORWARD, FFTW_ESTIMATE);
		        fftw_execute(plan);
				for(int i = 0; i < rows_; ++ i) {
					for(int j = 0; j < cols_; ++ j) {
						real_t temp = pow(mat_out[cols_ * i + j][0], 2) +
											pow(mat_out[cols_ * i + j][1], 2);
						min_val = (temp < min_val) ? temp : min_val;
						max_val = (temp > max_val) ? temp : max_val;
						int i_swap = (i + (rows_ >> 1)) % rows_;
						int j_swap = (j + (cols_ >> 1)) % cols_;
						img_data[i_swap * cols_ + j_swap] = temp;
					} // if j
				} // if i
    	    	fftw_destroy_plan(plan);
				fftw_free(mat_out);
				fftw_free(mat_in);
			#endif // USE_GPU
			/*for(unsigned int i = 0; i < rows_; ++ i) {
				for(unsigned int j = 0; j < cols_; ++ j) {
					real_t temp = (img_data[cols_ * i + j] - min_val) / (max_val - min_val);
					img.at<unsigned char>(i, j) = (unsigned char) 255 * temp;
					img_data[i * cols_ + j] = temp;
				} // for
			} // for
			// write it out
			cv::imwrite(HipRMCInput::instance().label() + "/base_fft_pattern.tif", img);*/
			wil::Image base_fft(rows_, cols_, 30, 30, 30);
			base_fft.construct_image(img_data);
			base_fft.save(HipRMCInput::instance().label() + "/base_fft_pattern.tif");
			#endif // USE_MODEL_INPUT

		#ifdef USE_MPI
		} // if
		#endif

		real_t *local_img_data = NULL;
		unsigned int *local_mask_data = NULL;

		#ifdef USE_MPI
			multi_node_.broadcast("real_world_masters", img_data, rows_ * cols_);
			multi_node_.broadcast("real_world_masters", mask_data, rows_ * cols_);

			std::cout << "++      Processor " << multi_node_.rank("world")
						<< " number of tiles: " << num_tiles_ << std::endl;

			// if a real_world master, distribute the matrices among all processors in real_world
			// we can either do a 1D partition, or a 2D partition. their disadvantages are:
			// 1D: less amount of parallelism, skinny matrices
			// 2D: some processors may be idle!
			// let us do 1D partitioning along rows for now.
			int real_world_size = multi_node_.size("real_world");
			int real_world_rank = multi_node_.rank("real_world");
			local_cols_ = cols_;
			local_rows_ = floor(rows_ / real_world_size) + (real_world_rank < rows_ % real_world_size);
			int prefix_row_sum = 0;
			multi_node_.scan_sum("real_world", local_rows_, prefix_row_sum);
			prefix_row_sum -= local_rows_;
			multi_node_.allgather("real_world", &prefix_row_sum, 1, row_offsets_, 1);
			// the masters will scatter the matrices among others in the real world
			// calculate sizes to send to all
			if(real_world_size > 1) {
        // TODO: check this case ...
				int *num_elements = NULL, *msg_sizes = NULL, *msg_displacements = NULL;
				if(multi_node_.is_master("real_world")) {
					num_elements = new (std::nothrow) int[real_world_size];
					msg_sizes = new (std::nothrow) int[real_world_size];
					msg_displacements = new (std::nothrow) int[real_world_size];
				} // if
				int local_elements = local_rows_ * local_cols_;		// number of elements
				multi_node_.gather("real_world", &local_elements, 1, num_elements, 1);
				if(multi_node_.is_master("real_world")) {
					msg_displacements[0] = 0;
					for(int i = 1; i < real_world_size; ++ i)
						msg_displacements[i] = msg_displacements[i - 1] + num_elements[i - 1];
				} // if
				local_img_data = new (std::nothrow) real_t[local_elements];
				local_mask_data = new (std::nothrow) unsigned int[local_elements];
				multi_node_.scatterv("real_world", img_data, num_elements, msg_displacements,
										local_img_data, local_elements);
				multi_node_.scatterv("real_world", mask_data, num_elements, msg_displacements,
										local_mask_data, local_elements);
				if(multi_node_.is_master("real_world")) {
					delete[] msg_displacements;
					delete[] num_elements;
					delete[] mask_data;
					delete[] img_data;
				} // if

				// compute offsets
				unsigned int local_size = local_rows_ * local_cols_;
				multi_node_.scan_sum("real_world", local_size, matrix_offset_);
				matrix_offset_ -= local_rows_ * local_cols_;
				unsigned int local_rows = local_rows_;
				multi_node_.scan_sum("real_world", local_rows, tile_offset_rows_);
				tile_offset_rows_ -= local_rows;
				tile_offset_cols_ = 0;

				// compute the local_tile_rows_ and local_tile_cols_
				local_tile_cols_ = tile_size_;		// all cols when distribution is along rows only
				// This is for the current case when current tile is not equally distributed
				// basically, compute with what you have
				int skip_rows = (rows_ - tile_size_) >> 1;
				int start_proc = 0, end_proc = 0;
				while(row_offsets_[start_proc] <= skip_rows && start_proc < real_world_size)
					++ start_proc;
				-- start_proc;		// the tile starts here
				while(row_offsets_[end_proc] <= skip_rows + tile_size_ && end_proc < real_world_size)
					++ end_proc;
				-- end_proc;		// tile ends here
				// all procs between start and end proc have tile rows same as local rows
				int my_rank = multi_node_.rank("real_world");
				std::cout << "START: " << start_proc << ", END: " << end_proc << std::endl;
				if(my_rank < start_proc || my_rank > end_proc) local_tile_rows_ = 0;
				else if(my_rank > start_proc && my_rank < end_proc) local_tile_rows_ = local_rows_;
				else if(my_rank == start_proc) {
					local_tile_rows_ = row_offsets_[start_proc] + local_rows_ - skip_rows;
				} else if(my_rank == end_proc) {
					local_tile_rows_ = rows_ - row_offsets_[end_proc] + - skip_rows;
				} else {
					std::cerr << "error: this is weird, an impossible case happened!" << std::endl;
					return false;
				} // if-else
			} else {
				local_img_data = img_data;
				local_mask_data = mask_data;
				matrix_offset_ = 0;
				tile_offset_rows_ = 0;
				tile_offset_cols_ = 0;
				local_tile_rows_ = tile_size_;
				local_tile_cols_ = tile_size_;
			} // if-else

  		in_pattern_.resize(local_rows_, local_cols_);
		  mask_mat_.resize(local_rows_, local_cols_);

	  	cropped_pattern_.resize(local_tile_rows_, local_tile_cols_);
  		cropped_mask_mat_.resize(local_tile_rows_, local_tile_cols_);
	  	vandermonde_mat_.resize(local_tile_rows_, local_tile_cols_);

		  in_pattern_.populate(local_img_data);
  		mask_mat_.populate(local_mask_data);

		#else // without MPI

			int local_rows_ = rows_;
			int local_cols_ = cols_;
			local_img_data = img_data;
			local_mask_data = mask_data;
			matrix_offset_ = 0;
			tile_offset_rows_ = 0;
			tile_offset_cols_ = 0;
			local_tile_rows_ = tile_size_;
			local_tile_cols_ = tile_size_;

      in_pattern_.resize(local_rows_, local_cols_);
		  mask_mat_.resize(local_rows_, local_cols_);

	  	cropped_pattern_.resize(local_tile_rows_, local_tile_cols_);
  		cropped_mask_mat_.resize(local_tile_rows_, local_tile_cols_);
	  	vandermonde_mat_.resize(local_tile_rows_, local_tile_cols_);

		  in_pattern_.populate(local_img_data);
  		mask_mat_.populate(local_mask_data);

		#endif // USE_MPI

		int tile_num_offset = 0;
		#ifdef USE_MPI
			// the assignment is round robin
			tile_num_offset = multi_node_.rank("world") % global_num_tiles_;
		#endif // USE_MPI

//		vec_uint_t indices;
//    if(HipRMCInput::instance().init_model().empty()) initialize_particles_random(indices);
//    else initialize_particles_image(indices);

		initialize_tiles(&(HipRMCInput::instance().loading()[tile_num_offset]),
                     HipRMCInput::instance().max_move_distance());

		std::cout << "++ Initialization done." << std::endl;

		delete[] local_mask_data;
		delete[] local_img_data;
		return true;
	} // RMC::init()


	// called once at RMC initialization
	bool RMC::initialize_tiles(const real_t* loading, unsigned int max_dist) {
		std::cout << "++ Initializing " << num_tiles_ << " tiles ... " << std::endl;
		unsigned int seeds[global_num_tiles_];
		woo::MTRandomNumberGenerator temp_rng;
		if(multi_node_.is_master("world")) {
			for(int i = 0; i < global_num_tiles_; ++ i)
				seeds[i] = floor(temp_rng.rand() * 1000000.0 * (i + 1.0));
		} // if
		// scatter the seeds - for now simply broadcasting
		multi_node_.broadcast("world", seeds, global_num_tiles_);
		unsigned int my_seed = seeds[multi_node_.rank("world")];
		
		// initialize tiles
		for(unsigned int i = 0; i < num_tiles_; ++ i) {
			vec_uint_t indices;
			if(HipRMCInput::instance().init_model().empty()) initialize_particles_random(i, indices);
			else initialize_particles_image(i, indices);
			tiles_.push_back(Tile(local_tile_rows_, local_tile_cols_, indices, size_,
								  i, my_seed
								  #ifdef USE_MPI
									, multi_node_.rank("world")
								  #endif
									));
		} // for
		for(unsigned int i = 0; i < num_tiles_; ++ i) {
			// construct prefix
			std::stringstream temp;
			#ifdef USE_MPI
				temp << std::setfill('0') << std::setw(4) << multi_node_.rank("world");
			#else
				temp << std::setfill('0') << std::setw(4);
			#endif
			temp << "_" << std::setfill('0') << std::setw(4) << i;
			char prefix[10];
			temp >> prefix;
			int num_particles = loading[i] * tile_size_ * tile_size_;
			tiles_[i].init(loading[i], max_dist, prefix, num_particles
					#ifdef USE_MPI
						, multi_node_
					#endif
					);
		} // for
		return true;
	} // RMC::initialize_tiles()


	// this is for every simulation set during scaling
	bool RMC::initialize_simulation(unsigned int scale_factor) {
		// scale pattern to current size
		//scale_pattern_to_tile(scale_factor);
		// do cropping and not scaling
		crop_pattern_to_tile(scale_factor);
		// process pattern, scale pixel intensities
		preprocess_pattern_and_mask(scale_factor);
		compute_base_norm();
		initialize_vandermonde(scale_factor);
		return true;
	} // RMC::initialize_simulation()


	// called at the beginning of each scale step
	bool RMC::initialize_simulation_tiles(int num_steps) {
		for(unsigned int i = 0; i < num_tiles_; ++ i) {
			tiles_[i].init_scale(base_norm_, cropped_pattern_, vandermonde_mat_,
									cropped_mask_mat_, num_steps
									#ifdef USE_MPI
										, multi_node_
									#endif
									);
		} // for
    /////// WARNING: THIS DOES NOT WORK WITH MPI YET!!!!!!!!!!!!!!! TODO ...
    if(global_num_tiles_ > 1 && !HipRMCInput::instance().independent()) {
      // get temperature and cooling from tile 0 (global index)
      real_t temperature = tiles_[0].temperature();
      real_t cooling = tiles_[0].cooling();
      for(int i = 1; i < num_tiles_; ++ i) {
        tiles_[i].temperature(temperature);
        tiles_[i].cooling(cooling);
      } // for
    } // if
		return true;
	} // RMC::initialize_simulation_tiles()


	bool RMC::destroy_simulation_tiles() {
		for(unsigned int i = 0; i < num_tiles_; ++ i) {
			tiles_[i].destroy_scale();
		} // for
		return true;
	} // RMC::destroy_simulation_tiles()


	// initialize vandermonde for current tile size
	bool RMC::initialize_vandermonde(unsigned int scale_fac) {
		// compute vandermonde matrix
		// generate 1st order power (full 360 deg rotation in polar coords)
		std::vector<complex_t> first_pow;
		for(unsigned int i = tile_offset_rows_; i < tile_offset_rows_ + local_tile_rows_; ++ i) {
			real_t temp = - 2.0 * PI_ * ((real_t) i / tile_size_);
			real_t temp_r = cos(temp);
			real_t temp_i = sin(temp);
			first_pow.push_back(complex_t(temp_r, temp_i));
		} // for
		if(vandermonde_mat_.num_rows() + scale_fac == local_tile_rows_) {
			vandermonde_mat_.incr_rows(scale_fac);
		} else if(vandermonde_mat_.num_rows() != local_tile_rows_) {
			std::cerr << "error: Mr. Vandermonde is very angry! "
						<< vandermonde_mat_.num_rows() << ", " << local_tile_rows_ << std::endl;
			return false;
		} // if-else
		if(vandermonde_mat_.num_cols() + scale_fac == local_tile_cols_) {
			vandermonde_mat_.incr_columns(scale_fac);
		} else if(vandermonde_mat_.num_cols() != tile_size_) {
			std::cerr << "error: Mr. Vandermonde is very angry! "
						<< vandermonde_mat_.num_cols() << ", " << local_tile_cols_ << std::endl;
			return false;
		} // if-else
		// initialize first column
		typename mat_complex_t::col_iterator citer = vandermonde_mat_.column(0);
		for(unsigned int i = 0; i < citer.size(); ++ i) citer[i] = complex_t(1.0, 0.0);
		// compute rest of the matrix
		typename mat_complex_t::col_iterator curr_citer = vandermonde_mat_.begin_col();
		typename mat_complex_t::col_iterator prev_citer = vandermonde_mat_.begin_col();
		++ curr_citer;
		for(; curr_citer != vandermonde_mat_.end_col(); ++ curr_citer, ++ prev_citer) {
			for(unsigned int i = 0; i < local_tile_rows_; ++ i)
				curr_citer[i] = prev_citer[i] * first_pow[i];
		} // while
		//print_cmatrix("vandermonde_mat", vandermonde_mat_.data(), tile_size_, tile_size_);
		return true;
	} // RMC::initialize_vandermonde()


	bool RMC::initialize_particles_random(unsigned int index, vec_uint_t &indices) {
		indices.clear();
		// create array of random indices
		unsigned int start = 0; //matrix_offset_ - local_tile_size_ * local_tile_size_;
		for(unsigned int i = start; i < tile_size_ * tile_size_; ++ i) indices.push_back(i);
		#ifdef USE_MPI
			// currently all procs will have the whole indices array
			if(multi_node_.is_master("real_world")) {
    #endif
        // using mersenne-twister
        // TODO: use woo library instead ... 
				std::mt19937_64 gen(time(NULL) * (index + 1)
                            #ifdef USE_MPI
                              + multi_node_.rank("world")
                            #endif
                           );
				std::shuffle(indices.begin(), indices.end(), gen);
    #ifdef USE_MPI
			} // if
			unsigned int *recv_buff = new (std::nothrow) unsigned int[indices.size()];
			multi_node_.broadcast("real_world", &indices[0], indices.size());
			delete[] recv_buff;
		#endif // USE_MPI
		//print_array("indices", (unsigned int*)&indices[0], indices.size());
		return true;
	} // RMC::initialize_particles_random()


  bool RMC::initialize_particles_image(unsigned int index, vec_uint_t &indices) {
    indices.clear();
    // read model image
    // warning: all procs are doing this ... improve TODO
		wil::Image model_img(0, 0, 30, 30, 30);
		if(!model_img.read_tiff(HipRMCInput::instance().init_model().c_str(), rows_, cols_)) return false;
		real_t *temp_model = new (std::nothrow) real_t[rows_ * cols_];
		model_img.get_data(temp_model);
    vec_uint_t empty;
		for(unsigned int i = 0; i < rows_ * cols_; ++ i) {
		  real_t temp = temp_model[i];
			if(255 * temp < 180) empty.push_back(i);
      else indices.push_back(i);
		} // for
    indices.insert(indices.end(), empty.begin(), empty.end());
		delete[] temp_model;
    return true;
  } // RMC::initialize_particles_image()


/*	bool RMC::scale_pattern_to_tile(unsigned int scale_factor) {
		if(size_ == tile_size_) {
			scaled_pattern_ = in_pattern_;
		} else {
			real_t* pdata = new (std::nothrow) real_t[size_ * size_];
			memcpy(pdata, in_pattern_.data(), size_ * size_ * sizeof(real_t));
			real_t* scaled_pdata = NULL;
			wil::scale_image((int) size_, (int) size_, (int) tile_size_, (int) tile_size_,
								pdata, scaled_pdata);
			// increase the size of the scaled pattern
			if(scaled_pattern_.num_rows() + scale_factor == tile_size_) {
				scaled_pattern_.incr_rows(scale_factor);
				scaled_pattern_.incr_columns(scale_factor);
			} // if
			// populate with the scaled data
			scaled_pattern_.populate(scaled_pdata);
			delete[] scaled_pdata;
		} // if-else
		return true;
	} // RMC::scale_pattern_to_tile()
*/

	// this constructs cropped pattern to a given tile_size_ (local_tile_rows_ and local_tile_cols_)
	bool RMC::crop_pattern_to_tile(unsigned int scale_factor) {
		#ifdef USE_MPI
			if(size_ == tile_size_) {	// this is the last step
        local_tile_rows_ = local_rows_;
        local_tile_cols_ = local_cols_;
				//if(local_tile_rows_ != local_rows_ || local_tile_cols_ != local_cols_) {
				//	std::cerr << "error: something fishy with local tile sizes" << std::endl;
				//	return false;
				//} // if
				cropped_pattern_ = in_pattern_;
				cropped_mask_mat_ = mask_mat_;
			} else {
				// compute local_tile_rows_ and local_tile_cols_
				local_tile_rows_ = local_tile_cols_ = 0;
				int total_skip_rows = (rows_ - tile_size_) >> 1;
				int pnum = multi_node_.size("real_world");
				int prank = multi_node_.rank("real_world");
				// find the processor rank which contains the start row (total_skip_rows)
				unsigned int proc_row_offsets[pnum + 1];
				multi_node_.allgather("real_world", &tile_offset_rows_, 1, proc_row_offsets, 1);
        proc_row_offsets[pnum] = proc_row_offsets[pnum - 1] + local_rows_;
				int p1 = 0; while(total_skip_rows >= proc_row_offsets[p1]) ++ p1;
				-- p1;	// one above
				// find proc rank which contains the end row (total_skip_rows + tile_size)
				int p2 = 0; while(total_skip_rows + tile_size_ >= proc_row_offsets[p2]) ++ p2;
				-- p2;	// one above
				local_tile_cols_ = tile_size_;
				if(prank == p1 && prank == p2) local_tile_rows_ = tile_size_;
				else if(prank == p1)
					local_tile_rows_ = proc_row_offsets[prank] + local_rows_ - total_skip_rows;
				else if(prank == p2)
					local_tile_rows_ = proc_row_offsets[prank] + local_rows_ -
										(total_skip_rows + tile_size_);
				else if(prank > p1 && prank < p2) local_tile_rows_ = local_rows_;
				else local_tile_rows_ = 0;

				// increase the size of the local cropped matrices
				int insert_rows = local_tile_rows_ - cropped_pattern_.num_rows();
				int insert_cols = local_tile_cols_ - cropped_pattern_.num_cols();
				cropped_pattern_.incr_rows(insert_rows);
				cropped_pattern_.incr_columns(insert_cols);
				cropped_mask_mat_.incr_rows(insert_rows);
				cropped_mask_mat_.incr_columns(insert_cols);
				if(cropped_pattern_.num_rows() != cropped_mask_mat_.num_rows() ||
						cropped_pattern_.num_cols() != cropped_mask_mat_.num_cols()) {
					std::cerr << "error: mismatch in sizes of cropped pattern and mask" << std::endl;
					return false;
				} // if

				// DO NOT REDISTRIBUTE, JUST COMPUTE ON WHAT YOU ALREADY HAVE:
				int skip_rows = (local_rows_ - local_tile_rows_) >> 1;
				int skip_cols = (size_ - tile_size_) >> 1;
				for(int i = 0; i < local_tile_rows_; ++ i) {
					for(int j = 0; j < local_tile_cols_; ++ j) {
						cropped_pattern_(i, j) = in_pattern_(skip_rows + i, skip_cols + j);
						cropped_mask_mat_(i, j) = mask_mat_(skip_rows + i, skip_cols + j);
					} // for
				} // for

				// TODO: REDISTRIBUTE THE TILE:
				/*// compute communication sizes
				int total_skip_rows = (rows_ - tile_size_) >> 1;
				int pnum = multi_node_.size("real_world");
				int prank = multi_node_.rank("real_world");
				int send_rows[pnum], recv_rows[pnum];
				// find the processor rank which contains the start row (total_skip_rows)
				int proc_row_offsets[pnum];
				multi_node_.allgather("real_world", &tile_offset_rows_, 1, proc_row_offsets, 1);
				int p1 = 0; while(total_skip_rows >= proc_row_offsets[p1]) ++ p1;
				-- p1;	// one above
				for(int i = 0; i < p1; ++ i) recv_rows[i] = 0;	// nothing to receive from these
				// find proc rank which contains the end row (total_skip_rows + tile_size)
				int p2 = 0; while(total_skip_rows + tile_size_ >= proc_row_offsets[p2]) ++ p2;
				-- p2;	// one above
				// calculate number of rows this proc owns that are to be sent elsewhere
				int rows_to_send = 0;
				if(prank < p1 || prank > p2) rows_to_send = 0;
				else if(p1 < prank && p2 > prank) rows_to_send = local_rows_;
				else if(p1 == p2 && prank == p1) rows_to_send = tile_size_;
				else if(p1 == prank) rows_to_send = proc_row_offsets[prank] + local_rows_ -
													total_skip_rows;
				else if(p2 == prank) rows_to_send = rows_ - total_skip_rows - tile_size_;
				else {
					std::cerr << "error: some impossible case?" << std::endl;
					return false;
				} // if-else
				// calculate the number of rows this proc will receive from others
				int rows_to_receive = local_tile_rows_;
				// calculate the number of rows to be sent to each processor
				int rows_to_send_procs[pnum];
				// calculate the number of rows to be received from each processor
				int rows_to_receive_procs[pnum];
				// ... */

			} // if-else
		#else // without MPI
			if(size_ == tile_size_) {
				cropped_pattern_ = in_pattern_;
				cropped_mask_mat_ = mask_mat_;
			} else {
				// increase the size of the cropped pattern
				if(cropped_pattern_.num_rows() + scale_factor == tile_size_) {
					cropped_pattern_.incr_rows(scale_factor);
					cropped_pattern_.incr_columns(scale_factor);
				} // if
				if(cropped_mask_mat_.num_rows() + scale_factor == tile_size_) {
					cropped_mask_mat_.incr_rows(scale_factor);
					cropped_mask_mat_.incr_columns(scale_factor);
				} // if
				// populate with the cropped data
				int skip_rows = (size_ - tile_size_) >> 1;
				int skip_cols = skip_rows;
				for(int i = 0; i < tile_size_; ++ i) {
					for(int j = 0; j < tile_size_; ++ j) {
						cropped_pattern_(i, j) = in_pattern_(skip_rows + i, skip_cols + j);
						cropped_mask_mat_(i, j) = mask_mat_(skip_rows + i, skip_cols + j);
					} // for j
				} // for i
			} // if-else
			local_tile_cols_ = tile_size_;
			local_tile_rows_ = tile_size_;
			real_t max_val = 0.0, min_val = 1e10;
			for(int i = 0; i < tile_size_; ++ i) {
				for(int j = 0; j < tile_size_; ++ j) {
					real_t temp = cropped_pattern_(i, j);
					min_val = (temp < min_val) ? temp : min_val;
					max_val = (temp > max_val) ? temp : max_val;
				} // for j
			} // for i
			// write them out (for verification)
			/*cv::Mat img(tile_size_, tile_size_, 0);
			for(unsigned int i = 0; i < tile_size_; ++ i) {
				for(unsigned int j = 0; j < tile_size_; ++ j) {
					real_t temp = (cropped_pattern_[tile_size_ * i + j] - min_val) / (max_val - min_val);
					img.at<unsigned char>(i, j) = (unsigned char) 255 * temp;
				} // for
			} // for
			cv::imwrite(HipRMCInput::instance().label() + "/cropped_pattern.tif", img);*/
			wil::Image img(tile_size_, tile_size_, 30, 30, 30);
			img.construct_image(cropped_pattern_.data());
			img.save(HipRMCInput::instance().label() + "/cropped_pattern.tif");
			/*for(unsigned int i = 0; i < tile_size_; ++ i) {
				for(unsigned int j = 0; j < tile_size_; ++ j) {
					unsigned int temp = cropped_mask_mat_[tile_size_ * i + j];
					img.at<unsigned char>(i, j) = (unsigned char) 255 * temp;
				} // for
			} // for
			cv::imwrite(HipRMCInput::instance().label() + "/cropped_mask.tif", img);*/
			wil::Image mask_img(tile_size_, tile_size_, 30, 30, 30);
			img.construct_image(cropped_mask_mat_.data());
			img.save(HipRMCInput::instance().label() + "/cropped_mask.tif");
		#endif
		return true;
	} // RMC::crop_pattern_to_tile()


	bool RMC::preprocess_pattern_and_mask(unsigned int scale_fac) {
		#ifdef USE_MPI
			// nothing to do here ...
		#else
			double min_val, max_val;
			//woo::matrix_min_max(scaled_pattern_, min_val, max_val);
			woo::matrix_min_max(cropped_pattern_, min_val, max_val);
			double threshold = min_val;// + 2 * ceil(max_val / (min_val + 1));
			//std::cout << "MIN: " << min_val << ", MAX: " << max_val
			//			<< ", THRESH: " << threshold << std::endl;
			// sanity check
			//if(scaled_pattern_.num_rows() != tile_size_) {
			if(cropped_pattern_.num_rows() != tile_size_) {
				std::cerr << "error: you are now really in grave danger: "
							<< cropped_pattern_.num_rows() << ", " << tile_size_ << std::endl;
							//<< scaled_pattern_.num_rows() << ", " << tile_size_ << std::endl;
				return false;
			} else {
				//std::cout << "be happiee: " << scaled_pattern_.num_rows() << ", " << tile_size_ << std::endl;
				//std::cout << "be happiee: " << cropped_pattern_.num_rows() << ", " << tile_size_ << std::endl;
			} // if-else
			// apply threshold and
			// scale pixel intensities to span all of 0 - 255
			// and generate mask_mat_
			//memset(mask_mat_, 0, tile_size_ * tile_size_ * sizeof(unsigned int));
			/*if(mask_mat_.num_rows() + scale_fac == tile_size_) {
				mask_mat_.incr_rows(scale_fac);
				mask_mat_.incr_columns(scale_fac);
			} else if(mask_mat_.num_rows() != tile_size_) {
				std::cerr << "error: you have a wrong mask. "
							<< mask_mat_.num_rows() << ", " << tile_size_ << std::endl;
				return false;
			} // if-else
			mask_mat_.fill(1);
			if(min_val < max_val) {
				for(unsigned int i = 0; i < tile_size_; ++ i) {
					for(unsigned int j = 0; j < tile_size_; ++ j) {
						double temp;
						//if(scaled_pattern_(i, j) > threshold) {
						//	temp = 255 * (scaled_pattern_(i, j) - threshold) / (max_val - threshold);
						if(cropped_pattern_(i, j) > threshold) {
				//			temp = (cropped_pattern_(i, j) - threshold) / (max_val - threshold);
						} else {
				//			temp = 0.0;
							mask_mat_(i, j) = 0;
						} // if-else
						//scaled_pattern_(i, j) = temp;
				//		cropped_pattern_(i, j) = temp;
					} // for
				} // for
			} // if*/

			//normalize_cropped_pattern();

			// normalize the cropped pattern
			/*cv::Mat img(tile_size_, tile_size_, 0);
			for(unsigned int i = 0; i < tile_size_; ++ i) {
				for(unsigned int j = 0; j < tile_size_; ++ j) {
					real_t temp = (cropped_pattern_[tile_size_ * i + j] - min_val) / (max_val - min_val);
					img.at<unsigned char>(i, j) = (unsigned char) 255 * temp;
				} // for
			} // for
			// write it out
			cv::imwrite(HipRMCInput::instance().label() + "/normalized_pattern.tif", img);*/
			wil::Image img(tile_size_, tile_size_, 30, 30, 30);
			img.construct_image(cropped_pattern_.data());
			img.save(HipRMCInput::instance().label() + "/normalized_pattern.tif");

			/*real_t * data = new (std::nothrow) real_t[tile_size_ * tile_size_];
			for(int i = 0; i < tile_size_; ++ i) {
				for(int j = 0; j < tile_size_; ++ j) {
					int i_swap = i;//(i + (size_ >> 1)) % size_;
					int j_swap = j;//(j + (size_ >> 1)) % size_;
					data[tile_size_ * i + j] = 255 * cropped_pattern_(i_swap, j_swap);
				} // for j
			} // for i
			wil::Image img2(tile_size_, tile_size_, 30, 30, 30);
			img2.construct_image(data);
			img2.save(HipRMCInput::instance().label() + "/my_normalized_pattern.tif");
			delete[] data;*/
		#endif // USE_MPI

		return true;
	} // RMC::preprocess_pattern_and_mask()

	
	bool RMC::compute_base_norm() {
		// compute base norm
		double base_norm = 0.0;
		#ifdef USE_MPI
			#pragma omp parallel shared(base_norm)
			{
				#pragma omp for collapse(2) reduction(+:base_norm)
				for(unsigned int i = 0; i < cropped_pattern_.num_rows(); ++ i) {
					for(unsigned int j = 0; j < cropped_pattern_.num_cols(); ++ j) {
						base_norm += cropped_pattern_(i, j) * cropped_mask_mat_(i, j);
					} // for
				} // for
			}
			multi_node_.allreduce_sum("real_world", base_norm, base_norm_);
		#else
			unsigned int maxi = tile_size_;
			#pragma omp parallel shared(base_norm)
			{
				#pragma omp for collapse(2) reduction(+:base_norm)
				for(unsigned int i = 0; i < maxi; ++ i) {
					for(unsigned int j = 0; j < maxi; ++ j) {
						base_norm += cropped_pattern_(i, j) * cropped_mask_mat_(i, j);
					} // for
				} // for
			}
			base_norm_ = base_norm;
		#endif // USE_MPI
		//std::cout << "++          Base pattern norm value: " << base_norm_ << std::endl;
		return true;
	} // RMC::compute_base_norm();


	/*bool RMC::initialize_mask() {
		// create mask and loading arays
		mytimer.start();
		in_mask_len_ = mask_len;
		in_mask_ = new (std::nothrow) unsigned int[mask_len];
		if(in_mask_ == NULL) return false;
		memcpy(in_mask_, mask, mask_len * sizeof(unsigned int));
		// generate mask matrix
		mask_mat_ = new (std::nothrow) unsigned int[size2];
		for(unsigned int i = 0; i < size2; ++ i) mask_mat_[i] = 1;
		for(unsigned int i = 0; i < in_mask_len_; ++ i) mask_mat_[in_mask_[i]] = 0;
		mytimer.stop();
		std::cout << "**** Mask creation time: " << mytimer.elapsed_msec() << " ms." << std::endl;
		//print_matrix("mask_mat:", mask_mat_, size_, size_);

		return true;
	} // RMC::initialize_mask()*/


	// simulate RMC
	bool RMC::simulate(int num_steps, unsigned int rate, unsigned int scale_factor = 1) {

		#ifdef USE_MPI
			if(multi_node_.is_idle("real_world")) return true;
			if(multi_node_.is_master("real_world")) {
		#endif
			std::cout << std::endl << "++                Current tile size: "
						<< tile_size_ << " x " << tile_size_ << std::endl;
		#ifdef USE_MPI
			} // if
		#endif

		if(!initialize_simulation(scale_factor)) {
			std::cerr << "error: failed to initialize simulation set" << std::endl;
			return false;
		} // if
		if(!initialize_simulation_tiles(num_steps)) {
			std::cerr << "error: failed to initialize simulation set" << std::endl;
			return false;
		} // if

		//std::cout << "++ Saving initial images ..." << std::endl;
		//for(unsigned int i = 0; i < num_tiles_; ++ i) {
		//	#ifdef USE_GPU
		//		tiles_[i].update_f_mats();
		//	#endif
		//	tiles_[i].save_mat_image(i);
		//	//tiles_[i].save_fmat_image(i);
		//	tiles_[i].save_mat_image_direct(i);
		//} // for
		unsigned int ten_percent = floor(num_steps / 10);
		unsigned int curr_percent = 10;
		std::cout << "++ Performing simulation on " << num_tiles_ << " tiles ... ";
		#ifdef USE_MPI
			std::cout << "[says process " << multi_node_.rank("world") << "]";
			multi_node_.barrier("world");
		#endif
		std::cout << std::endl;
		woo::BoostChronoTimer sim_timer;
		sim_timer.start();
		for(unsigned int step = 0; step < num_steps; ++ step) {
			if((step + 1) % ten_percent == 0) {
				#ifdef USE_MPI
					if(multi_node_.is_master("world"))
				#endif
						std::cout << "    " << curr_percent << "\% done at step " << step + 1
									<< std::endl;
				curr_percent += 10;
			} // if
			for(unsigned int i = 0; i < num_tiles_; ++ i) {
				tiles_[i].simulate_step(cropped_pattern_, vandermonde_mat_, cropped_mask_mat_,
										base_norm_, step, num_steps
										#ifdef USE_MPI
											, multi_node_
										#endif
										);
				if((step + 1) % rate == 0) tiles_[i].update_model(
										#ifdef USE_MPI
											multi_node_
										#endif
										);
				/*if(step % 100 == 0) {
					tiles_[i].update_model();
					#ifdef USE_GPU
						tiles_[i].update_f_mats();
					#endif
					tiles_[i].save_mat_image((step / 100 + 1));
					tiles_[i].save_mat_image_direct(step / 100 + 1);	// save a_mat
				} // if*/
			} // for
		} // for
		sim_timer.stop();
		#ifdef USE_MPI
			multi_node_.barrier("world");
			if(multi_node_.is_master("world"))
		#endif
				std::cout << "++ Simulation done [" << sim_timer.elapsed_msec() << " ms]" << std::endl;
		for(unsigned int i = 0; i < num_tiles_; ++ i) {
			tiles_[i].print_new_times();
		} // for

		for(unsigned int i = 0; i < num_tiles_; ++ i) {
			double chi2 = 0.0;
			tiles_[i].finalize_result(chi2
									#ifdef USE_MPI
										, multi_node_
									#endif
									);
			std::cout << "++ ";
			#ifdef USE_MPI
				std::cout << "P" << multi_node_.rank("world");
			#else
				std::cout << "  ";
			#endif
			std::cout << " Tile " << i << " final chi2-error value: " << chi2 << std::endl;
			std::cout << "++ ";
			#ifdef USE_MPI
				std::cout << "  P" << multi_node_.rank("world");
			#else
				std::cout << "    ";
			#endif
			std::cout << " Tile " << i << " total accepted moves: "
						<< tiles_[i].accepted_moves()
						<< " [" << (real_t) tiles_[i].accepted_moves() / num_steps * 100 << "%]"
						<< std::endl;

			#ifdef USE_GPU
				// temp ... for bandwidth computation of dft2 ...
				unsigned int num_blocks = ceil((real_t) tile_size_ / CUDA_BLOCK_SIZE_X_) *
											ceil((real_t) tile_size_ / CUDA_BLOCK_SIZE_Y_);
				unsigned int read_bytes = num_blocks * (CUDA_BLOCK_SIZE_X_ + CUDA_BLOCK_SIZE_Y_) *
											sizeof(cucomplex_t);
				unsigned int write_bytes = num_blocks * CUDA_BLOCK_SIZE_X_ * CUDA_BLOCK_SIZE_Y_ *
											sizeof(cucomplex_t);
				std::cout << "++ DFT2 Bandwidth: "
						<< (float) (read_bytes + write_bytes) * num_steps * 1000 /
							(tiles_[i].dft2_time() * 1024 * 1024)
						<< " MB/s" << std::endl;
			#endif

			//std::cout << "++ Saving model ... ";
			#ifdef USE_MPI
			if(multi_node_.is_master("real_world"))
			#endif
				tiles_[i].save_model();
			tiles_[i].clear_chi2_list();
			//std::cout << "done." << std::endl << std::endl;
		} // for
		//for(unsigned int i = 0; i < num_tiles_; ++ i) {
		//	tiles_[i].print_times();
		//} // for
    #ifdef USE_MPI
		multi_node_.barrier("world");
    #endif
		//std::cout << "P" << multi_node_.rank() << ": SIM SIM SIM!!!!" << std::endl;
    destroy_simulation_tiles();

		return true;
	} // RMC::simulate()


	bool RMC::simulate_and_scale() {
		#ifdef USE_MPI
			if(multi_node_.is_idle("world")) return true;
			if(multi_node_.is_master("world")) {
		#endif
				std::cout << std::endl << "++ Performing scaled simulation ..." << std::endl;
		#ifdef USE_MPI
			} // if
		#endif
		int num_steps_fac = HipRMCInput::instance().num_steps_factor();
		unsigned int scale_factor = HipRMCInput::instance().scale_factor();
		unsigned int rate = 10000;		// FIXME ... hardcoded ... remove
		unsigned int num_steps = num_steps_fac * tile_size_;
		unsigned int curr_scale_fac = scale_factor;
		woo::BoostChronoTimer sim_timer;
		sim_timer.start();
		simulate(num_steps, rate, scale_factor);
		for(int tsize = tile_size_, iter = 0; tsize < size_; tsize += scale_factor, ++ iter) {
			if(tile_size_ < size_) {
				// loop over the local tiles
				for(int i = 0; i < num_tiles_; ++ i) {
		      curr_scale_fac = scale_factor;
					tiles_[i].update_model(
										#ifdef USE_MPI
											multi_node_
										#endif
										);
					if(tile_size_ + scale_factor > size_) {
						curr_scale_fac = size_ - tile_size_;
					} // if
					for(int s = 0; s < curr_scale_fac; ++ s) {
						tiles_[i].scale_step();
					} // for
					if(tiles_[i].size() != tile_size_ + curr_scale_fac) {
						std::cerr << "error: you are in graaaaaaave danger!" << std::endl;
						return false;
					} // if
				} // for
				tile_size_ += curr_scale_fac;
			} // if
			num_steps = num_steps_fac * tile_size_;
			simulate(num_steps, rate, curr_scale_fac);
		} // for
		#ifdef USE_MPI
		multi_node_.barrier("world");
		#endif
		sim_timer.stop();
		#ifdef USE_MPI
			if(multi_node_.is_master("real_world"))
		#endif
				std::cout << "**           Total simulation time: " << sim_timer.elapsed_msec()
							<< " ms." << std::endl;
		return true;
	} // RMC::simulate_and_scale()


	// this is for testing scaling
	// it scaled in_pattern_ itself
	bool RMC::scale(unsigned int final_size) {
		unsigned int num_steps = final_size - size_;
		for(unsigned int i = 0; i < num_steps; ++ i) {
			// make sure all indices info is in a_mat_
			tiles_[0].update_model(
								#ifdef USE_MPI
									multi_node_
								#endif
								);
			tiles_[0].scale_step();
			//tiles_[0].save_mat_image_direct(i);
			// update indices_ and other stuff using the new model
			tiles_[0].update_from_model();
		} // for

		return true;
	} // RMC::scale()

} // namespace hir
