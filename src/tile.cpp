/***
  *  Project: HipRMC - High Performance Reverse Monte Carlo Simulations
  *
  *  File: tile.cpp
  *  Created: Jan 25, 2013
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifdef USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif // USE_GPU
#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP

#include <map>
#include "woo/timer/woo_boostchronotimers.hpp"

#include "tile.hpp"
#include "constants.hpp"

namespace hir {

	// constructor
	Tile::Tile(unsigned int rows, unsigned int cols, const std::vector<unsigned int>& indices,
				unsigned int final_size, unsigned int index) :
		a_mat_(std::max(rows, cols), std::max(rows, cols)),
		#ifndef USE_DFT
			virtual_a_mat_(rows, cols),
		#endif
		diff_mat_(rows, cols),
		f_mat_i_(0),
		mod_f_mat_i_(0),
		indices_(indices),
		dft_mat_(rows, cols),
		mt_rand_gen_(time(NULL) * (index + 1)),
		#ifndef USE_GPU
			autotuner_(rows, cols, indices),
		#endif
		fft_update_time_(0.0), reduction_time_(0.0), misc_time_(0.0), mpi_time_(0.0)
		{

		woo::BoostChronoTimer mytimer;

		size_ = std::max(rows, cols);
		final_size_ = final_size;
		rows_ = rows;
		cols_ = cols;

		tstar_ = 1.0;
		cooling_factor_ = 0.0;
		tstar_set_ = false;

		// two buffers each
		mytimer.start();
		f_mat_.push_back(mat_complex_t(rows_, cols_));
		f_mat_.push_back(mat_complex_t(rows_, cols_));
		mytimer.stop();
		//std::cout << "**   FFT matrix initialization time: " << mytimer.elapsed_msec() << " ms."
		//			<< std::endl;
		mytimer.start();
		mod_f_mat_.push_back(mat_real_t(rows_, cols_));
		mod_f_mat_.push_back(mat_real_t(rows_, cols_));
		mytimer.stop();
		//std::cout << "**      FFT mod initialization time: " << mytimer.elapsed_msec() << " ms."
		//			<< std::endl;
		#ifdef USE_GPU
			// device memory allocation takes all the time
			mytimer.start();
			unsigned int size2 = final_size_ * final_size_;
			cucomplex_buff_ = new (std::nothrow) cucomplex_t[size2];
			mytimer.stop();
		#endif // USE_GPU
		//std::cout << "**       Memory initialization time: " << mytimer.elapsed_msec()
		//				<< " ms." << std::endl;
	} // Tile::Tile()


	// copy constructor
	Tile::Tile(const Tile& tile):
		size_(tile.size_),
		rows_(tile.rows_),
		cols_(tile.cols_),
		final_size_(tile.final_size_),
		a_mat_(tile.a_mat_),
		min_row_index_(tile.min_row_index_),
		max_row_index_(tile.max_row_index_),
		f_mat_(tile.f_mat_),
		mod_f_mat_(tile.mod_f_mat_),
		indices_(tile.indices_),
		#ifndef USE_DFT
			virtual_a_mat_(tile.virtual_a_mat_),
		#endif
		diff_mat_(tile.diff_mat_),
		f_mat_i_(tile.f_mat_i_),
		mod_f_mat_i_(tile.mod_f_mat_i_),
		loading_factor_(tile.loading_factor_),
		tstar_(tile.tstar_),
		cooling_factor_(tile.cooling_factor_),
		tstar_set_(tile.tstar_set_),
		num_particles_(tile.num_particles_),
		max_move_distance_(tile.max_move_distance_),
		//model_norm_(tile.model_norm_),
		//c_factor_(tile.c_factor_),
		dft_mat_(tile.dft_mat_),
		prev_chi2_(tile.prev_chi2_),
		old_pos_(tile.old_pos_),
		new_pos_(tile.new_pos_),
		old_index_(tile.old_index_),
		new_index_(tile.new_index_),
		mt_rand_gen_(time(NULL)),
		#ifndef USE_GPU
			autotuner_(tile.autotuner_),
		#endif
		fft_update_time_(tile.fft_update_time_),
		reduction_time_(tile.reduction_time_),
		misc_time_(tile.misc_time_),
		mpi_time_(tile.mpi_time_)
		{
		#ifdef USE_GPU
			unsigned int size2 = final_size_ * final_size_;
			cucomplex_buff_ = new (std::nothrow) cucomplex_t[size2];
		#endif // USE_GPU
	} // Tile::Tile()


	Tile::~Tile() {
		#ifdef USE_GPU
			delete[] cucomplex_buff_;
		#endif // USE_GPU
	} // Tile::~Tile()


	// initialize with raw data - done only once
	bool Tile::init(real_t loading, unsigned int max_move_dist, char* prefix, int num_particles
			#ifdef USE_MPI
				, woo::MultiNode& multi_node
			#endif
			) {
		woo::BoostChronoTimer mytimer;

		loading_factor_ = loading;
		//tstar_ = tstar;
		//cooling_factor_ = cooling;
		tstar_set_ = false;
		max_move_distance_ = max_move_dist;
		num_particles_ = num_particles;
		// NOTE: the first num_particles_ entries in indices_ are filled, rest are empty
		prefix_ = std::string(prefix);

		// fill a_mat_ with particles
		//std::cout << "UPDATING MODEL" << std::endl;
		mytimer.start();
		update_model(
					#ifdef USE_MPI
						multi_node
					#endif
					);
		#ifdef USE_GPU
			unsigned int block_x = CUDA_BLOCK_SIZE_X_;
			unsigned int block_y = CUDA_BLOCK_SIZE_Y_;
			gtile_.init(a_mat_.data(), final_size_, size_, block_x, block_y);
		#endif // USE_GPU
		mytimer.stop();
		std::cout << "**  Initial model construction time: " << mytimer.elapsed_msec() << " ms."
					<< std::endl;
		std::cout << "++              Number of particles: " << num_particles_ << std::endl;
		misc_time_ += mytimer.elapsed_msec();

		return true;
	} // Tile::init()


	// to be executed at simulation beginning after a scaling
	bool Tile::init_scale(real_t base_norm, mat_real_t& pattern, mat_complex_t& vandermonde,
							mat_uint_t& mask, int num_steps
							#ifdef USE_MPI
								, woo::MultiNode& multi_node
							#endif
							) {
    // FIXME: cannot remember why I had the following check in scaling ...
		if(pattern.num_rows() != rows_ || vandermonde.num_rows() != rows_ ||
				mask.num_rows() != rows_ || pattern.num_cols() != cols_ ||\
				vandermonde.num_cols() != cols_ || mask.num_cols() != cols_) {
			std::cerr << "error: some matrix size is not what it should be! "
                << rows_ << " " << pattern.num_rows() << " " << a_mat_.num_rows() << std::endl;
			return false;
		} // if

		// reset all timers to 0
		vmove_time_ = dft2_time_ = mod_time_ = norm_time_ = chi2_time_ = rest_time_ = 0.0;

		// compute fft of a_mat_ into fft_mat_ and other stuff
		woo::BoostChronoTimer mytimer;
		mytimer.start();
		#ifdef USE_GPU
			unsigned int block_x = CUDA_BLOCK_SIZE_X_;
			unsigned int block_y = CUDA_BLOCK_SIZE_Y_;
			for(unsigned int i = 0; i < size_; ++ i) {
				for(unsigned int j = 0; j < size_; ++ j) {
					complex_t temp = vandermonde(i, j);
					cucomplex_buff_[size_ * i + j].x = temp.real();
					cucomplex_buff_[size_ * i + j].y = temp.imag();
				} // for
			} // for
			gtile_.init_scale(pattern.data(), cucomplex_buff_, a_mat_.data(), mask.data(),
								size_, block_x, block_y);
		#else
			fft_in_ = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size_ * size_);
			fft_out_ = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size_ * size_);
			fft_plan_ = fftw_plan_dft_2d(size_, size_, fft_in_, fft_out_, FFTW_FORWARD, FFTW_ESTIMATE);
		#endif // USE_GPU

		// compute row offsets
		#ifdef USE_MPI
		  row_offsets_[0] = 0;
			int local_rows = rows_;
			int prefix_sums = 0;
			multi_node.scan_sum("real_world", local_rows, prefix_sums);
			prefix_sums -= rows_;
			multi_node.allgather("real_world", &prefix_sums, 1, row_offsets_, 1);
		#endif // USE_MPI
		mytimer.stop();
		misc_time_ += mytimer.elapsed_msec();

		// autotune temperature (tstar)
		#ifndef USE_GPU		// currently gpu version does not have autotuning
			mytimer.start();
			if(!autotune_temperature(pattern, vandermonde, mask, base_norm, num_steps
					#ifdef USE_MPI
						, multi_node
					#endif
					)) {
				std::cerr << "error: failed to autotune temperature" << std::endl;
				return false;
			} // if
			mytimer.stop();
//			tstar_ = 1.0;
//			cooling_factor_ = 1.0;
			//std::cout << "TEMPERATURE = " << tstar_ << std::endl;
			//std::cout << "COOLING = " << cooling_factor_ << std::endl;
			std::cout << "**      Temperature autotuning time: " << mytimer.elapsed_msec()
						<< " ms." << std::endl;
			tstar_set_ = true;
			misc_time_ += mytimer.elapsed_msec();
		#endif // USE_GPU

		#ifdef USE_MPI
			mytimer.start();
			compute_fft_mat(multi_node);
			compute_mod_mat(f_mat_i_, multi_node);
			copy_mod_mat(1 - mod_f_mat_i_);
			mytimer.stop(); fft_update_time_ += mytimer.elapsed_msec();
			mytimer.start();
			prev_chi2_ = compute_chi2(pattern, mod_f_mat_[1 - mod_f_mat_i_], mask,
										base_norm, multi_node);
			mytimer.stop(); reduction_time_ += mytimer.elapsed_msec();
		#else
			mytimer.start();
			compute_fft_mat();
			compute_mod_mat(f_mat_i_);
			copy_mod_mat(1 - mod_f_mat_i_);
			mytimer.stop(); fft_update_time_ += mytimer.elapsed_msec();
			mytimer.start();
			prev_chi2_ = compute_chi2(pattern, mod_f_mat_[1 - mod_f_mat_i_], mask, base_norm);
			mytimer.stop(); reduction_time_ += mytimer.elapsed_msec();
		#endif // USE_MPI
		//std::cout << "++         Initial chi2-error value: " << prev_chi2_ << std::endl;

		accepted_moves_ = 0;

		return true;
	} // Tile::init_scale()


	bool Tile::destroy_scale() {
		//std::cout << "****** destroying scale tiles" << std::endl;
		#ifdef USE_GPU
			gtile_.destroy_scale();
		#else	// use CPU
			fftw_destroy_plan(fft_plan_);
			fftw_free(fft_out_);
			fft_out_ = NULL;
			fftw_free(fft_in_);
			fft_in_ = NULL;
		#endif
		return true;
	} // Tile::destroy_scale()


	bool Tile::copy_mod_mat(unsigned int src_i) {
		#ifdef USE_GPU
			gtile_.copy_mod_mat(src_i);
		#else
			mod_f_mat_[1 - src_i].populate(mod_f_mat_[src_i].data());
		#endif // USE_GPU
		return true;
	} // Tile::copy_mod_mat()


	// in case of gpu version, this assumes all data is already on the gpu
	bool Tile::simulate_step(const mat_real_t& pattern,
							mat_complex_t& vandermonde,
							const mat_uint_t& mask,
							real_t base_norm, unsigned int iter, unsigned int max_iter
							#ifdef USE_MPI
								, woo::MultiNode& multi_node
							#endif
							) {
		// do all computations in scratch buffers
		unsigned int f_scratch_i = 1 - f_mat_i_;
		unsigned int mod_f_scratch_i = 1 - mod_f_mat_i_;

		//std::cout << "HAHAHAHAHAHAHA: " << multi_node.rank("real_world") << "\t" << indices_.size() << std::endl;

    #ifdef USE_MPI
		multi_node.barrier("real_world");
    #endif

		mytimer_.start();
		virtual_move_random_particle_restricted(max_move_distance_
				#ifdef USE_MPI
					, multi_node
				#endif
				);
		mytimer_.stop(); vmove_time_ += mytimer_.elapsed_msec();
		misc_time_ += mytimer_.elapsed_msec();

		//std::cout << "HEHEHEHEHEHE: " << multi_node.rank("real_world") << std::endl;

		mytimer_.start();
    #ifdef USE_MPI
		multi_node.barrier("real_world");
    #endif
		mytimer_.stop(); mpi_time_ += mytimer_.elapsed_msec();

		mytimer_.start();
		#ifdef USE_DFT
			unsigned int old_row = old_index_ / size_;
			unsigned int old_col = old_index_ % size_;
			unsigned int new_row = new_index_ / size_;
			unsigned int new_col = new_index_ % size_;
			compute_dft2(vandermonde, old_row, old_col, new_row, new_col, dft_mat_
					#ifdef USE_MPI
						, multi_node
					#endif
					);
			#ifndef USE_GPU
				update_fft_mat(dft_mat_, f_mat_[f_mat_i_], f_mat_[f_scratch_i]);
			#endif // USE_GPU
		#else	// use FFT at all steps
			#ifdef USE_MPI
				update_virtual_model(multi_node);
				compute_fft_mat(f_scratch_i, multi_node);
			#else
				update_virtual_model();
				compute_fft_mat(f_scratch_i);
			#endif
		#endif
		mytimer_.stop(); dft2_time_ += mytimer_.elapsed_msec();
		fft_update_time_ += mytimer_.elapsed_msec();

    #ifdef USE_MPI
		mytimer_.start();
		multi_node.barrier("real_world");
		mytimer_.stop(); mpi_time_ += mytimer_.elapsed_msec();
    #endif
		//std::cout << "HIHIHIHIHIHIHIHHI: " << multi_node.rank("real_world") << std::endl;

		mytimer_.start();
		#ifdef USE_MPI
			compute_mod_mat(f_scratch_i, multi_node);
		#else
			compute_mod_mat(f_scratch_i);
		#endif
		mytimer_.stop(); mod_time_ += mytimer_.elapsed_msec();
		//fft_update_time_ += mytimer_.elapsed_msec();

		mytimer_.start();
		#ifdef USE_MPI
		multi_node.barrier("real_world");
		#endif
		mytimer_.stop(); mpi_time_ += mytimer_.elapsed_msec();

		//mytimer_.start();
		//compute_model_norm(mod_f_scratch_i, mask);
		//double new_c_factor = base_norm / model_norm_;
		//mytimer_.stop(); norm_time_ += mytimer_.elapsed_msec();

		mytimer_.start();
		double new_chi2 = compute_chi2(pattern, mod_f_mat_[mod_f_scratch_i], mask, base_norm
									#ifdef USE_MPI
										, multi_node
									#endif
									);
		chi2_list_.push_back(new_chi2);		// save this chi2 value
		mytimer_.stop(); chi2_time_ += mytimer_.elapsed_msec();
		reduction_time_ += mytimer_.elapsed_msec();

		mytimer_.start();
		#ifdef USE_MPI
		multi_node.barrier("real_world");
    #endif
		mytimer_.stop(); mpi_time_ += mytimer_.elapsed_msec();
		//std::cout << "HUHUHUHUHUHUHU: " << multi_node.rank("real_world") << std::endl;

		mytimer_.start();
		bool accept = false;
		#ifdef USE_MPI
			if(multi_node.is_master("real_world")) {
		#endif
        accept = accept_reject(prev_chi2_, new_chi2, tstar_, cooling_factor_, iter, max_iter);
		#ifdef USE_MPI
			} // if
			mytimer2_.start();
      int temp_accept = (accept) ? 1 : 0;
			multi_node.broadcast("real_world", &temp_accept, 1);
			multi_node.barrier("real_world");
      accept = (temp_accept == 1);
			mytimer2_.stop(); mpi_time_ += mytimer2_.elapsed_msec();
		#endif

		if(accept) {	// accept the move
			// update to newly computed stuff
			// make scratch as current
			++ accepted_moves_;
			#ifdef USE_MPI
				move_particle(new_chi2, multi_node);
			#else
				move_particle(new_chi2);
			#endif
		} // if
		mytimer_.stop(); rest_time_ += mytimer_.elapsed_msec();
		misc_time_ += mytimer_.elapsed_msec();

		// write current model at every "steps"
		if(iter % 10000 == 0) {
			#ifdef USE_MPI
				update_model(multi_node);
				//std::cout << "SHITTY SHIT SHIT SHIT SHIT " << multi_node.rank("real_world") << std::endl;
				if(multi_node.is_master("real_world")) create_image("model", iter / 1000, a_mat_, false);
				//std::cout << "CRAPPY CRAP CRAP CRAP CRAP " << multi_node.rank("real_world") << std::endl;
			#else
				update_model();
				create_image("model", iter / 1000, a_mat_, false);
			#endif
		} // if

		mytimer_.start();
		#ifdef USE_MPI
		multi_node.barrier("real_world");
    #endif
		mytimer_.stop(); mpi_time_ += mytimer_.elapsed_msec();

		//std::cout << "HERERERERERE: " << multi_node.rank("real_world") << std::endl;

		return true;
	} // Tile::simulate_step()


  bool Tile::accept_reject(real_t prev_chi2, real_t new_chi2, real_t tstar, real_t cooling,
                           unsigned int iter, unsigned int max_iter) {
    double diff_chi2 = prev_chi2 - new_chi2;
    bool accept = false;
    if(diff_chi2 > 0.0) accept = true;
    else {
      real_t temperature = 0.0, p = 0.0;
      if(tstar < 1e-30) {
        temperature = 0.0; p = 0.0;
      } else {
        //temperature = tstar / (1.0 + cooling * (iter / max_iter));
        temperature = tstar / ((2.0 - tstar) * (1.0 + cooling * iter));
        //p = exp((diff_chi2 / temperature) * (pow((double)max_iter, 10) / 1e14));  // works best when scaling
        //p = exp((diff_chi2 / temperature) * (pow((double)max_iter, 20) / 1e38));  // works best when scaling
        real_t multiplier = pow(max_iter, 5);
        p = exp((diff_chi2 / temperature) * multiplier);  // works best when scaling
      } // if-else
      real_t prand = mt_rand_gen_.rand();
      //std::cout << "**** prev: " << prev_chi2 << " new: " << new_chi2 << " diff: " << diff_chi2
      //          << " p: " << p << " prand: " << prand << std::endl;
      if(prand < p) accept = true;
    } // if-else
    return accept;
  } // Tile::accept_reject()


	void Tile::create_image(std::string str, unsigned int iter, const mat_real_t &mat, bool swapped) {
		std::stringstream num_iter;
		num_iter << std::setfill('0') << std::setw(4) << iter;
		char str0[5];
		num_iter >> str0;
		std::stringstream num_size;
		num_size << std::setfill('0') << std::setw(4) << size_;
		char str1[5];
		num_size >> str1;
		double min_val, max_val;
		woo::matrix_min_max(mat, min_val, max_val);

		/*cv::Mat img(size_, size_, 0);
		for(unsigned int i = 0; i < size_; ++ i) {
			for(unsigned int j = 0; j < size_; ++ j) {
				int i_swap = i, j_swap = j;
				if(swapped) {
					i_swap = (i + (size_ >> 1)) % size_;
					j_swap = (j + (size_ >> 1)) % size_;
				}
				img.at<unsigned char>(i, j) = (unsigned char) 255 * 
												((mat(i_swap, j_swap) - min_val) / (max_val - min_val));
			} // for
		} // for
		// write it out
		cv::imwrite(HipRMCInput::instance().label() + "/" + std::string(str0) + "_" + str + ".tif", img);
		*/
		int rows = mat.num_rows();
		int cols = mat.num_cols();
		//std::cout << "====================== SO I REACHED HERE: " << rows << ", " << cols << std::endl;
		real_t * data = new (std::nothrow) real_t[rows * cols];
		//real_t * data = (real_t*) malloc(rows * cols * sizeof(real_t));
		//real_t data[rows * cols];
		if(data == NULL) {
			std::cerr << "error: failed to allocate memory for image data" << std::endl;
			exit(1);
		} // if
		//std::cout << "SO I REACHED HERERE =====================" << std::endl;
		for(int i = 0; i < rows; ++ i) {
			for(int j = 0; j < cols; ++ j) {
				int i_swap = i, j_swap = j;
				// FIXME do quadrant swap thingy
				//if(swapped) {
				//	i_swap = (i + (rows >> 1)) % rows;
				//	j_swap = (j + (cols >> 1)) % cols;
				//} // if
				data[cols * i + j] = 255 * ((mat(i_swap, j_swap) - min_val) / (max_val - min_val));
			} // for j
		} // for i
		wil::Image img(rows, cols, 30, 30, 30);
		img.construct_image(data);
		std::string filename = prefix_ + "_" + std::string(str1) + "_" + std::string(str0) +
								"_" + str + ".tif";
		img.save(HipRMCInput::instance().label() + "/" + filename);
		delete[] data;
		//free(data);
	} // Tile::create_image()


	bool Tile::save_chi2_list() {
		std::stringstream num_size;
		num_size << std::setfill('0') << std::setw(4) << size_;
		char str1[5];
		num_size >> str1;
		std::string filename = prefix_ + "_" + std::string(str1) + "_chi2_list.dat";
		std::ofstream chi2out(HipRMCInput::instance().label() + "/" + filename, std::ios::out);
		for(int i = 0; i < chi2_list_.size(); ++ i) {
			chi2out << i << "\t" << std::setprecision(std::numeric_limits<double>::digits10 + 1)
					<< chi2_list_[i] << std::endl;
		} // for
		chi2out.close();
		return true;
	} // Tile::save_chi2_list()


	bool Tile::clear_chi2_list() {
		chi2_list_.clear();
		return true;
	} // Tile::clear_chi2_list()


	bool Tile::update_model(
			#ifdef USE_MPI
				woo::MultiNode& multi_node
			#endif
			) {
		return update_a_mat();
	} // Tile::update_model()


	#ifndef USE_DFT

	bool Tile::update_virtual_model() {
		unsigned int x, y;
		virtual_a_mat_.fill(0.0);
		#pragma omp parallel for
		for(unsigned int i = 0; i < num_particles_; ++ i) {
			x = indices_[i] / size_;
			y = indices_[i] % size_;
			virtual_a_mat_(x, y) = 1.0;
		} // for
		x = old_index_ / size_;
		y = old_index_ % size_;
		if(virtual_a_mat_(x, y) != 1.0) std::cout << "OMGOMGOMGOMGOMG" << std::endl;
		virtual_a_mat_(x, y) = 0.0;
		x = new_index_ / size_;
		y = new_index_ % size_;
		if(virtual_a_mat_(x, y) != 0.0) std::cout << "LOLLOLLOLLOLLOL" << std::endl;
		virtual_a_mat_(x, y) = 1.0;
		#ifdef USE_GPU
			gtile_.copy_virtual_model(virtual_a_mat_);
		#endif
		return true;
	} // Tile::update_virtual_model()

	#endif


	bool Tile::update_from_model() {
		update_indices();
		return true;
	} // Tile::update_from_model()


	#ifndef USE_GPU // use CPU

	bool Tile::compute_fft_mat(
			#ifdef USE_MPI
				woo::MultiNode& multi_node
			#endif
			) {
		unsigned int size2 = size_ * size_;
		real_t* orig_a_mat = a_mat_.data();
		for(int i = 0; i < size2; ++ i) {
			fft_in_[i][0] = a_mat_[i];
			fft_in_[i][1] = 0.0;
			fft_out_[i][0] = 0.0;
			fft_out_[i][1] = 0.0;
		} // for
		// execute fft
		fftw_execute(fft_plan_);
		#pragma omp parallel
		{
			#pragma omp for collapse(2)
			for(unsigned int i = 0; i < size_; ++ i) {
				for(unsigned int j = 0; j < size_; ++ j) {
					f_mat_[f_mat_i_](i, j) = complex_t(fft_out_[size_ * i + j][0],
														fft_out_[size_ * i + j][1]);
				} // for
			} // for
		} // omp parallel

		return true;
	} // Tile::compute_fft_mat()


	bool Tile::compute_fft(const mat_real_t& src, mat_complex_t& dst) {
		unsigned int size2 = size_ * size_;
		for(int i = 0; i < size2; ++ i) {
			fft_in_[i][0] = src[i];
			fft_in_[i][1] = 0.0;
			fft_out_[i][0] = 0.0;
			fft_out_[i][1] = 0.0;
		} // for
		// execute fft
		fftw_execute(fft_plan_);
		#pragma omp parallel
		{
			#pragma omp for collapse(2)
			for(unsigned int i = 0; i < size_; ++ i) {
				for(unsigned int j = 0; j < size_; ++ j) {
					dst(i, j) = complex_t(fft_out_[size_ * i + j][0], fft_out_[size_ * i + j][1]);
				} // for
			} // for
		} // omp parallel

		return true;
	} // Tile::compute_fft()


	#ifndef USE_DFT

	bool Tile::compute_fft_mat(unsigned int buff_i
			#ifdef USE_MPI
				, woo::MultiNode& multi_node
			#endif
			) {
		unsigned int size2 = size_ * size_;
		for(int i = 0; i < size2; ++ i) {
			fft_in_[i][0] = virtual_a_mat_[i];
			fft_in_[i][1] = 0.0;
			fft_out_[i][0] = 0.0;
			fft_out_[i][1] = 0.0;
		} // for
		// execute fft
		fftw_execute(fft_plan_);

		#pragma omp parallel
		{
			#pragma omp for collapse(2)
			for(unsigned int i = 0; i < size_; ++ i) {
				for(unsigned int j = 0; j < size_; ++ j) {
					f_mat_[buff_i](i, j) = complex_t(fft_out_[size_ * i + j][0], fft_out_[size_ * i + j][1]);
				} // for
			} // for
		} // omp parallel

		return true;
	} // Tile::compute_fft_mat()

	#endif	// USE_DFT


	bool Tile::execute_fftw(fftw_complex* input, fftw_complex* output) {
		// create fft plan
		fftw_plan plan;
		plan = fftw_plan_dft_2d(rows_, cols_, input, output, FFTW_FORWARD, FFTW_ESTIMATE);
		fftw_execute(plan);
		// destroy fft plan
		fftw_destroy_plan(plan);
		return true;
	} // Tile::execute_cufft()

	#else // USE_GPU

	bool Tile::compute_fft_mat(
			#ifdef USE_MPI
				woo::MultiNode& multi_node
			#endif
			) {
		gtile_.compute_fft_mat(f_mat_i_);
		return true;
	} // Tile::compute_fft_mat()


	bool Tile::compute_fft(const mat_real_t& src, mat_complex_t& dst) {
//		gtile_.compute_fft(src, dst);
		return true;
	} // Tile::compute_fft()


	#ifndef USE_DFT

	bool Tile::compute_fft_mat(unsigned int buff_i) {
		gtile_.compute_virtual_fft_mat(buff_i);
		return true;
	} // Tile::compute_fft_mat()

	#endif // USE_DFT

	#endif // USE_GPU


	bool Tile::compute_mod_mat(unsigned int f_i
			#ifdef USE_MPI
				, woo::MultiNode& multi_node
			#endif
			) {
		#ifdef USE_GPU
			mytimer2_.start();
			gtile_.compute_mod_mat(f_i, 1 - mod_f_mat_i_);
			mytimer2_.stop(); fft_update_time_ += mytimer2_.elapsed_msec();
			mytimer2_.start();
			gtile_.normalize_mod_mat(1 - mod_f_mat_i_);
			mytimer2_.stop(); reduction_time_ += mytimer2_.elapsed_msec();
		#else // USE CPU
			#pragma omp parallel for collapse(2)
			for(unsigned int i = 0; i < rows_; ++ i) {
				for(unsigned int j = 0; j < cols_; ++ j) {
					complex_t temp_f = f_mat_[f_i](i, j);
					real_t temp = temp_f.real() * temp_f.real() + temp_f.imag() * temp_f.imag();
					mod_f_mat_[1 - mod_f_mat_i_](i, j) = temp;
				} // for
			} // for
			//normalize_mod_mat(1 - mod_f_mat_i_);
			normalize(mod_f_mat_[1 - mod_f_mat_i_]);
			return true;
		#endif // USE_GPU
	} // Tile::compute_mod_mat()


	bool Tile::compute_mod(const mat_complex_t& src, mat_real_t& dst) {
		#ifdef USE_GPU
//			gtile_.compute_mod(src, dst);
//			gtile_.normalize(dst);
		#else // USE CPU
			#pragma omp parallel for collapse(2)
			for(unsigned int i = 0; i < size_; ++ i) {
				for(unsigned int j = 0; j < size_; ++ j) {
					complex_t temp_f = src(i, j);
					real_t temp = temp_f.real() * temp_f.real() + temp_f.imag() * temp_f.imag();
					dst(i, j) = temp;
				} // for
			} // for
			normalize(dst);
		#endif // USE_GPU

		return true;
	} // Tile::compute_mod()


	/*bool Tile::normalize_mod(unsigned int mat_i) {
		real_t sum = 0.0;
		for(int i = 0; i < size_; ++ i) {
			for(int j = 0; j < size_; ++ j) {
				if(i == 0 && j == 0) continue;
				sum += mod_f_mat_[mat_i](i, j);
			} // for j
		} // for i
		real_t avg = sum / (size_ * size_);
		for(int i = 0; i < size_; ++ i) {
			for(int j = 0; j < size_; ++ j) {
				if(i == 0 && j == 0) continue;
				mod_f_mat_[mat_i](i, j) /= avg;
			} // for j
		} // for i

		return true;
	} // Tile::normalize_mod()*/


	/*bool Tile::normalize_mod_mat(unsigned int mat_i) {
		real_t min_val, max_val;
		woo::matrix_min_max(mod_f_mat_[mat_i], min_val, max_val);
		//min_val = max_val = mod_f_mat_[mat_i](1, 1);
		//for(unsigned int i = 0; i < size_; ++ i) {
		//	for(unsigned int j = 0; j < size_; ++ j) {
		//		if(i == 0 && j == 0) continue;
		//		real_t temp = mod_f_mat_[mat_i](i, j);
		//		min_val = (temp < min_val) ? temp : min_val;
		//		max_val = (temp > max_val) ? temp : max_val;
		//	} // for
		//} // for
		#pragma omp parallel for collapse(2)
		for(unsigned int i = 0; i < size_; ++ i) {
			for(unsigned int j = 0; j < size_; ++ j) {
				if(i == 0 && j == 0) continue;
				mod_f_mat_[mat_i](i, j) = (mod_f_mat_[mat_i](i, j) - min_val) / (max_val - min_val);
			} // for
		} // for
		return true;
	} // Tile::normalize_mod_mat()*/


	#ifndef USE_GPU
	bool Tile::normalize(mat_real_t& mat) {
		real_t min_val, max_val;
		woo::matrix_min_max(mat, min_val, max_val);
		#pragma omp parallel for collapse(2)
		for(unsigned int i = 0; i < mat.num_rows(); ++ i) {
			for(unsigned int j = 0; j < mat.num_cols(); ++ j) {
				if(i == 0 && j == 0) continue;
				mat(i, j) = (mat(i, j) - min_val) / (max_val - min_val);
			} // for
		} // for
		return true;
	} // Tile::normalize()
	#endif


	/*bool Tile::compute_model_norm(unsigned int buff_i, const mat_uint_t& mask) {
		double model_norm = 0.0;
		unsigned int maxi = size_;// >> 1;
		#ifdef USE_GPU
			model_norm = gtile_.compute_model_norm(buff_i);
		#else
			#pragma omp parallel shared(model_norm)
			{
				#pragma omp for collapse(2) reduction(+:model_norm)
				for(unsigned int i = 0; i < maxi; ++ i) {
					for(unsigned int j = 0; j < maxi; ++ j) {
						if(i == 0 && j == 0) continue;
						int i_swap = (i + (size_ >> 1)) % size_;
						int j_swap = (j + (size_ >> 1)) % size_;
						model_norm += mod_f_mat_[buff_i](i, j) * mask(i_swap, j_swap);
					} // for
				} // for
			}
		#endif // USE_GPU
		model_norm_ = model_norm;
//		std::cout << "++++ model_norm: " << model_norm_ << std::endl;
		return true;
	} // Tile::compute_model_norm()*/


	//double Tile::compute_chi2(const mat_real_t& pattern, unsigned int mod_f_i, const mat_uint_t& mask,
	//							real_t c_factor, real_t base_norm) {
	/*real_t Tile::compute_chi2(const mat_real_t& pattern, unsigned int mod_f_i, const mat_uint_t& mask,
								real_t base_norm) {
		real_t chi2 = 0.0;
		#ifdef USE_GPU
			chi2 = gtile_.compute_chi2(mod_f_i, c_factor, base_norm);
		#else
			#pragma omp parallel for collapse(2) reduction(+:chi2)
			for(unsigned int i = 0; i < size_; ++ i) {
				for(unsigned int j = 0; j < size_; ++ j) {
					if(i == 0 && j == 0) continue;
					int i_swap = (i + (size_ >> 1)) % size_;
					int j_swap = (j + (size_ >> 1)) % size_;
					real_t temp = fabs(pattern(i_swap, j_swap) - mod_f_mat_[mod_f_i](i, j)) * mask(i_swap, j_swap);
					//real_t temp = fabs(pattern(i_swap, j_swap) - 2.5 * mod_f_mat_[mod_f_i](i, j));
					//real_t temp = fabs(pattern(i_swap, j_swap) - mod_f_mat_[mod_f_i](i, j) * c_factor);
					//diff_mat_(i, j) = temp;
					//std::cout << "--------- pattern: " << pattern(i_swap, j_swap)
					//			<< ", mod_f: " << mod_f_mat_[mod_f_i](i, j)
					//			<< ", c_factor: " << c_factor
					//			<< ", mod2: " << c_factor * mod_f_mat_[mod_f_i](i, j)
					//			<< ", chi: " << temp
					//			<< ", chi2: " << temp * temp << std::endl;
					chi2 += temp * temp;
					//if(pattern(i_swap, j_swap) != 0.0)
					//	chi2 += temp * temp / fabs(pattern(i_swap, j_swap));
				} // for
			} // for
		#endif // USE_GPU
		// normalize with something ... norm for now
		//chi2 = 1e10 * chi2 / (base_norm * base_norm);	// with 128x128
		//chi2 = 2e7 * chi2 / (base_norm * base_norm);	// with 32x32
		//chi2 = (pow((real_t) size_, 5) * 1e-6) * chi2 / (base_norm * base_norm);
		chi2 = (pow((real_t) size_, 2.5)) * chi2 / (base_norm * base_norm);
		return chi2;
	} // Tile::compute_chi2()*/


	real_t Tile::compute_chi2(const mat_real_t& a, const mat_real_t& b, const mat_uint_t& mask,
								real_t base_norm
								#ifdef USE_MPI
									, woo::MultiNode& multi_node
								#endif
								) {
		real_t chi2 = 0.0;
		#ifdef USE_GPU
//			chi2 = gtile_.compute_chi2(a, b);
			chi2 = gtile_.compute_chi2(mod_f_mat_i_, 0.0, base_norm);
		#else
			#pragma omp parallel for collapse(2) reduction(+:chi2)
			for(unsigned int i = 0; i < a.num_rows(); ++ i) {
				for(unsigned int j = 0; j < a.num_cols(); ++ j) {
					if(i == 0 && j == 0) continue;
					// FIXME: quadrant swap thingy ...
					int i_swap = (i + (size_ >> 1)) % size_;
					int j_swap = (j + (size_ >> 1)) % size_;
					//real_t temp = fabs(a(i_swap, j_swap) - b(i, j)) * mask(i_swap, j_swap);
					//chi2 += temp * temp;
//					real_t temp = fabs(a(i_swap, j_swap) * a(i_swap, j_swap) - b(i, j) * b(i, j))
//                        * mask(i_swap, j_swap);
          real_t logb = b(i, j);
          //if(b(i, j) > 1e-30) logb = log(b(i, j));
					real_t temp = fabs(a(i_swap, j_swap) * a(i_swap, j_swap) - logb * logb)
                        * mask(i_swap, j_swap);
					chi2 += temp;
				} // for
			} // for
		#endif // USE_GPU
		real_t total_chi2 = chi2;
		#ifdef USE_MPI
			multi_node.allreduce_sum("real_world", chi2, total_chi2);
		#endif
		//total_chi2 = (pow((real_t) size_, 2.5)) * total_chi2 / (base_norm * base_norm);
		total_chi2 = total_chi2 / (base_norm * base_norm);
		return total_chi2;
	} // Tile::compute_chi2()


	/*double Tile::compute_chi2(mat_real_t& pattern, unsigned int mod_f_i, real_t c_factor) {
		double chi2 = 0.0;
		//#ifdef USE_GPU
		//	chi2 = gtile_.compute_chi2(mod_f_i, c_factor);
		//#else
		//	#pragma omp parallel for collapse(2) reduction(+:chi2)
			unsigned int psize = pattern.num_rows();
			real_t* pdata = new (std::nothrow) real_t[psize * psize];
			memcpy(pdata, pattern.data(), psize * psize * sizeof(real_t));
			real_t* spdata;
			wil::scale_image((int)psize, (int)psize, (int)size_, (int)size_, pdata, spdata);
			for(unsigned int i = 0; i < size_; ++ i) {
				for(unsigned int j = 0; j < size_; ++ j) {
					real_t temp = spdata[size_ * i + j] - mod_f_mat_[mod_f_i](i, j) * c_factor;
					chi2 += temp * temp;
				} // for
			} // for
			//delete[] spdata;
		//#endif // USE_GPU
		return chi2;
	} // Tile::compute_chi2()*/


	// NOTE: NOT USED
	bool Tile::virtual_move_random_particle() {
		old_pos_ = floor(mt_rand_gen_.rand() * num_particles_);
		new_pos_ = floor(mt_rand_gen_.rand() * (size_ * size_ - num_particles_)) + num_particles_;
		old_index_ = indices_[old_pos_];
		new_index_ = indices_[new_pos_];
		//std::cout << "++++ old_pos,new_pos: " << old_pos_ << "," << new_pos_
		//			<< ", old_index,new_index: " << old_index_ << "," << new_index_ << std::endl;
		return true;
	} // Tile::virtual_move_random_particle()


	bool Tile::virtual_move_random_particle_restricted(unsigned int dist
			#ifdef USE_MPI
				, woo::MultiNode& multi_node
			#endif
			) {
		#ifdef USE_MPI
			int pos[4];
			if(multi_node.is_master("real_world")) {
		#endif
        //mt_rand_gen_.reset(time(NULL));
				while(1) {
					old_pos_ = floor(mt_rand_gen_.rand() * num_particles_);
					new_pos_ = floor(mt_rand_gen_.rand() *	(size_ * size_ - num_particles_)) +
								num_particles_;
					// FIXME: currently this assumes that all indices_ array is on all procs
					if(old_pos_ < indices_.size()) old_index_ = indices_[old_pos_];
					else {
						std::cerr << "errrrrooorrrrrr: " << old_pos_ << std::endl;
						return false;
					} // if-else
					if(new_pos_ < indices_.size()) new_index_ = indices_[new_pos_];
					else {
						std::cerr << "errrroorr: " << new_pos_ << std::endl;
						return false;
					} // if-else
					int old_x = old_index_ / size_, old_y = old_index_ % size_;
					int new_x = new_index_ / size_, new_y = new_index_ % size_;
					if((fabs(new_x - old_x) < dist || (size_ - fabs(new_x - old_x)) < dist) &&
							(fabs(new_y - old_y) < dist || (size_ - fabs(new_y - old_y)) < dist))
						break;
				} // while
		#ifdef USE_MPI
				pos[0] = old_pos_; pos[1] = new_pos_;
				pos[2] = old_index_; pos[3] = new_index_;
			} // if
			multi_node.broadcast("real_world", pos, 4);
			if(!multi_node.is_master("real_world")) {
				old_pos_ = pos[0]; new_pos_ = pos[1];
				old_index_ = pos[2]; new_index_ = pos[3];
			} // if
			//std::cout << "P" << multi_node.rank("real_world") << ": " << old_pos_ << ", "
			//			<< new_pos_ << ", " << old_index_ << ", " << new_index_ << std::endl;
		#endif // USE_MPI

		//std::cout << "++++ old_pos,new_pos: " << old_pos_ << "," << new_pos_
		//			<< ", old_index,new_index: " << old_index_ << "," << new_index_ << std::endl;
		return true;
	} // Tile::virtual_move_random_particle()


	bool Tile::move_particle(real_t new_chi2
			#ifdef USE_MPI
				, woo::MultiNode& multi_node
			#endif
			) {
		// to swap buffers
		f_mat_i_ = 1 - f_mat_i_;
		mod_f_mat_i_ = 1 - mod_f_mat_i_;
		prev_chi2_ = new_chi2;

		// to swap indices (moving the particle)
		indices_[old_pos_] = new_index_;
		indices_[new_pos_] = old_index_;

		return true;
	} // Tile::move_particle()


	#ifdef USE_DFT

	bool Tile::compute_dft2(mat_complex_t& vandermonde_mat, unsigned int old_row, unsigned int old_col,
							unsigned int new_row, unsigned int new_col, mat_complex_t& dft_mat
							#ifdef USE_MPI
								, woo::MultiNode& multi_node
							#endif
							) {
		// in current row-wise decomposition, each proc will have the required columns,
		// but need to find the the rows and communicate relevant parts to all
		// each row will be fully contained in the proc that owns it

		complex_t *old_row_data = new complex_t[cols_];
		complex_t *new_row_data = new complex_t[cols_];

		#ifdef USE_MPI
			int proc_id = 0;
			while(row_offsets_[proc_id] <= old_row && proc_id < multi_node.size("real_world"))
				++ proc_id;
			-- proc_id;
			if(multi_node.rank("real_world") == proc_id) {
		#endif
				// construct send buffer
				typename mat_complex_t::row_iterator iter1 = vandermonde_mat.row(old_row
																#ifdef USE_MPI
																	- row_offsets_[proc_id]
																#endif
																);
				for(int i = 0; i < cols_; ++ i) old_row_data[i] = iter1[i];
		#ifdef USE_MPI
			} // if
			multi_node.broadcast("real_world", old_row_data, cols_, proc_id);
			proc_id = 0;
			while(row_offsets_[proc_id] <= new_row && proc_id < multi_node.size("real_world"))
				++ proc_id;
			-- proc_id;
			if(multi_node.rank("real_world") == proc_id) {
		#endif
				// construct send buffer
				typename mat_complex_t::row_iterator iter2 = vandermonde_mat.row(new_row
																#ifdef USE_MPI
																	- row_offsets_[proc_id]
																#endif
																);
				for(int i = 0; i < cols_; ++ i) new_row_data[i] = iter2[i];
		#ifdef USE_MPI
			} // if
			multi_node.broadcast("real_world", new_row_data, cols_, proc_id);
		#endif
		#ifdef USE_GPU
//			gtile_.compute_dft2(old_row, old_col, new_row, new_col, num_particles_, dft_mat);
			gtile_.compute_dft2(old_row, old_col, new_row, new_col, num_particles_, f_mat_i_, f_mat_i_);
		#else
			//typename mat_complex_t::row_iterator old_row_iter = vandermonde_mat.row(old_row);
			typename mat_complex_t::col_iterator old_col_iter = vandermonde_mat.column(old_col);
			//typename mat_complex_t::row_iterator new_row_iter = vandermonde_mat.row(new_row);
			typename mat_complex_t::col_iterator new_col_iter = vandermonde_mat.column(new_col);
			#pragma omp parallel for collapse(2)
			for(int row = 0; row < rows_; ++ row) {
				for(int col = 0; col < cols_; ++ col) {
					complex_t new_temp = new_col_iter[row] * new_row_data[col];
					complex_t old_temp = old_col_iter[row] * old_row_data[col];
					// FIXME: fix size of the dft_mat: it has to be transpose!!!!!!
					//dft_mat(col, row) = (new_temp - old_temp);
					dft_mat(row, col) = (new_temp - old_temp);
				} // for
			} // for
		#endif // USE_GPU

		delete[] old_row_data;
		delete[] new_row_data;

		return true;
	} // Tile::compute_dft2()


	bool Tile::update_fft_mat(const mat_complex_t& dft_mat, const mat_complex_t& in_f_mat,
								mat_complex_t& out_f_mat) {
		#ifdef USE_GPU
			// this has been merged into compute_dft2 for gpu
		#else
			#pragma omp parallel for collapse(2)
			for(int i = 0; i < rows_; ++ i) {
				for(int j = 0; j < cols_; ++ j) {
					out_f_mat(i, j) = dft_mat(i, j) + in_f_mat(i, j);
				} // for j
			} // for i
		#endif // USE_GPU
		return true;
	} // Tile::update_fft_mat()

	/*bool Tile::update_fft(mat_complex_t& f_mat, mat_complex_t& dft_mat) {
		#ifdef USE_GPU
			// this has been merged into compute_dft2 for gpu
		#else
			#pragma omp parallel for collapse(2)
			for(int i = 0; i < size_; ++ i) {
				for(int j = 0; j < size_; ++ j) {
					f_mat(i, j) = dft_mat(i, j) + f_mat(i, j);
				} // for j
			} // for i
		#endif // USE_GPU
		return true;
	} // Tile::update_fft()*/

	#endif


/*	bool Tile::mask_mat(const mat_uint_t& mask_mat, unsigned int buff_i) {
		#ifdef USE_GPU
			/// this has been merged into compute_mod_mat for gpu
		#else
			#pragma omp parallel for collapse(2) shared(mask_mat)
			for(unsigned int i = 0; i < size_; ++ i) {
				for(unsigned int j = 0; j < size_; ++ j) {
					mod_f_mat_[buff_i](i, j) *= mask_mat(i, j);
				} // for
			} // for
		#endif // USE_GPU
		return true;
	} // Tile::mask_mat()
*/

	bool Tile::finalize_result(double& chi2
			#ifdef USE_MPI
				, woo::MultiNode& multi_node
			#endif
			) {
		// populate a_mat_
		#ifdef USE_MPI
			update_model(multi_node);
		#else
			update_model();
		#endif
		#ifdef USE_GPU
			// also copy f mat data to main memory from GPU
			update_f_mats();
		#endif
		chi2 = prev_chi2_;
		return true;
	} // Tile::finalize()


	// update a_mat_ using indices_
	bool Tile::update_a_mat() {
		a_mat_.fill(0.0);
		#pragma omp parallel for
		for(unsigned int i = 0; i < num_particles_; ++ i) {
			unsigned int x = indices_[i] / size_;
			unsigned int y = indices_[i] % size_;
			a_mat_(x, y) = 1.0;
		} // for

		// also copy to GPU ... temporary?
		#ifdef USE_GPU
			gtile_.copy_model(a_mat_);
		#endif
		return true;
	} // Tile::update_a_mat()


#ifdef USE_GPU
	// update the f and modf matrices on the host with data from device
	bool Tile::update_f_mats() {
		unsigned int size2 = size_ * size_;
		real_t *mod_f_buff = new (std::nothrow) real_t[size2];
		complex_t *f_buff = new (std::nothrow) complex_t[size2];
		if(mod_f_buff == NULL || f_buff == NULL) {
			std::cerr << "error: could not allocate memory for f buffers in update_f_mats" << std::endl;
			return false;
		} // if

		gtile_.copy_f_mats_to_host(cucomplex_buff_, mod_f_buff, f_mat_i_, mod_f_mat_i_);
		for(unsigned int i = 0; i < size2; ++ i)
			f_buff[i] = complex_t(cucomplex_buff_[i].x, cucomplex_buff_[i].y);
		f_mat_[f_mat_i_].populate(f_buff);
		mod_f_mat_[mod_f_mat_i_].populate(mod_f_buff);

		delete[] f_buff;
		delete[] mod_f_buff;
		return true;
	} // Tile::update_f_mats()
#endif


	// update indices, num_particles and loading_factor using a_mat_
	bool Tile::update_indices() {
		unsigned int rows = a_mat_.num_rows();
		unsigned int cols = a_mat_.num_cols();
		indices_.clear();
		std::vector<unsigned int> zero_indices;
		num_particles_ = 0;
		for(unsigned int i = 0; i < rows * cols; ++ i) {
			if(a_mat_[i] > 0.5) { indices_.push_back(i); ++ num_particles_; }
			else zero_indices.push_back(i);
		} // for
		indices_.insert(indices_.end(), zero_indices.begin(), zero_indices.end());
		//loading_factor_ = (real_t) num_particles_ / (rows * cols);
		//std::cout << "+++++++++++++++ actual loading: " << (real_t) num_particles_ / (rows * cols)
		//			<< std::endl;
		return true;
	} // Tile::update_indices()


    /*bool Tile::initialize_random_model() {
		indices_.clear();
		// create array of random indices
		for(unsigned int i = 0; i < tile_size_ * tile_size_; ++ i) indices_.push_back(i);
		// using mersenne-twister
		std::random_device rd;
		std::mt19937_64 gen(rd());
		std::shuffle(indices_.begin(), indices_.end(), gen);
		return true;
	} // Tile::initialize_random_model()*/


	bool Tile::print_times() {
		std::cout << "**               Particle move time: " << vmove_time_ << " ms." << std::endl;
		std::cout << "**                 DFT compute time: " << dft2_time_  << " ms." << std::endl;
		std::cout << "**          Mod matrix compute time: " << mod_time_   << " ms." << std::endl;
		std::cout << "**          Model norm compute time: " << norm_time_  << " ms." << std::endl;
		std::cout << "**          Chi2-error compute time: " << chi2_time_  << " ms." << std::endl;
		std::cout << "**                       Other time: " << rest_time_  << " ms." << std::endl;
		return true;
	} // Tile::print_times()


	bool Tile::print_new_times() {
		std::cout << "@@                  FFT update time: " << fft_update_time_ << " ms." << std::endl;
		std::cout << "@@                   Reduction time: " << reduction_time_  << " ms." << std::endl;
		std::cout << "@@               Miscellaneous time: " << misc_time_   << " ms." << std::endl;
		std::cout << "@@           MPI communication time: " << mpi_time_  << " ms." << std::endl;
		return true;
	} // Tile::print_new_times()


	bool Tile::print_a_mat() {
		//print_matrix("a_mat", a_mat_.data(), a_mat_.num_rows(), a_mat_.num_cols());
		return true;
	} // Tile::print_a_mat()


} // namespace hir
