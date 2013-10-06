/***
  *  Project: WOO Libraries
  *
  *  File: reduce.cuh
  *  Created: Feb 12, 2013
  *  Modified: Sun 25 Aug 2013 09:12:11 AM PDT
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

//#include <nvToolsExt.h>

namespace woo {
namespace cuda {

	const unsigned int MAX_CUDA_THREADS_		= 1024;
	const unsigned int MAX_GRID_SIZE_			= 65535;	// max number of blocks
	const unsigned int BLOCK_DIM_REDUCE_TEST_	= 256;	// number of threads - assuming power of 2
	const unsigned int BLOCK_DIM_REDUCE_		= 256;	// number of threads - assuming power of 2
	const unsigned int BLOCK_DIM_REDUCE_ONE_	= 256;	// number of threads - assuming power of 2
	const unsigned int NUM_SUBTILES_			= 8;	// number of subtiles in a block
														// (processed by one thread block)

	extern __shared__ unsigned char d_data_sh[];

	/* a test imlementation: reversed loop of original - performs OK */

	template <typename data_t, typename functor_t>
	__global__
	void reduce_block_rev(const data_t* __restrict__ input, const unsigned int n, const data_t init,
							functor_t op, data_t* output) {
		__const__ unsigned int num_subtiles = NUM_SUBTILES_;
		unsigned int input_i = num_subtiles * blockDim.x * blockIdx.x + threadIdx.x;

		// start with init
		data_t sum = init;
		if(input_i < n) sum = input[input_i];

		// reduce the subtiles into single set (one sum for each thread)
		// this is the sequential component
		input_i += blockDim.x;
		for(unsigned int i = 1; input_i < n && i < num_subtiles; ++ i, input_i += blockDim.x) {
			sum = op(sum, input[input_i]);
		} // for
		
		// latter half threads store their data to shared memory
		// first half reduce those to their registers
		// reduce in log_2(blockDim.x) steps
		data_t *data_sh = ((data_t*) d_data_sh);
		unsigned int len = blockDim.x >> 1;
		while(len > 0) {
			if(threadIdx.x >= len && threadIdx.x < len << 1) data_sh[threadIdx.x] = sum;
			__syncthreads();
			if(threadIdx.x < len) sum = op(sum, data_sh[len + threadIdx.x]);
			len = len >> 1;
		} // while

		// write reduced output to global memory
		if(threadIdx.x == 0) output[blockIdx.x] = sum;
	} // reduce_block()


	template <typename iterator_t, typename data_t, typename functor_t>
	data_t reduce_rev(iterator_t start, iterator_t end, data_t init, functor_t op) {

		const unsigned int num_subtiles = NUM_SUBTILES_;
		const unsigned int block_size = BLOCK_DIM_REDUCE_TEST_;
		const unsigned int tile_size = block_size * num_subtiles;
		const unsigned int n_grid_max = MAX_GRID_SIZE_ * tile_size;
		const unsigned int n = end - start;
		const unsigned int num_grids = ceil((float) n / n_grid_max);

		unsigned int d_shmem_size = block_size * sizeof(data_t);
		unsigned int output_size = MAX_GRID_SIZE_;

		//nvtxRangeId_t nvtx0 = nvtxRangeStart("rev_setup");
		data_t *output[num_grids];
		data_t *base_output[num_grids];
		unsigned int n_grid[num_grids];
		unsigned int num_blocks[num_grids];
		data_t *input[num_grids];
		cudaStream_t stream[num_grids];
		bool to_break[num_grids];
		for(unsigned int grid = 0; grid < num_grids; ++ grid) {	// for each grid
			cudaMalloc((void**) &output[grid], output_size * sizeof(data_t));
			base_output[grid] = output[grid];
			// number of data elements to process in this grid:
			n_grid[grid] = n_grid_max;
			// the input data segment to process
			input[grid] = start + grid * n_grid_max;
			cudaStreamCreate(&stream[grid]);
			to_break[grid] = false;
		} // for
		n_grid[num_grids - 1] = n - (num_grids - 1) * n_grid_max;
		//nvtxRangeEnd(nvtx0);

		while(1) {
			for(unsigned int grid = 0; grid < num_grids; ++ grid) {	// for each grid
				if(!to_break[grid]) {
					num_blocks[grid] = ceil((float) n_grid[grid] / tile_size);
					cudaStreamSynchronize(stream[grid]);
					//nvtxRangeId_t nvtx1 = nvtxRangeStart("rev_kernel");
					reduce_block_rev <<< num_blocks[grid], block_size, d_shmem_size, stream[grid] >>>
										(input[grid], n_grid[grid], init, op, output[grid]);
					//nvtxRangeEnd(nvtx1);
					cudaError_t err = cudaGetLastError();
					if(err != cudaSuccess) {
						std::cerr << "error: something went wrong in kernel launch: "
									<< cudaGetErrorString(err) << std::endl;
					} // if
				} // if
			} // for
			bool to_break_all = true;
			for(unsigned int grid = 0; grid < num_grids; ++ grid) {	// for each grid
				if(!to_break[grid]) {
					n_grid[grid] = num_blocks[grid];
					if(num_blocks[grid] == 1) { to_break[grid] = true; continue; }
					to_break_all = false;
					data_t *temp = input[grid]; input[grid] = output[grid]; output[grid] = temp;
				} // if
			} // for
			if(to_break_all) break;
		} // while

		//nvtxRangeId_t nvtx2 = nvtxRangeStart("rev_finalize");
		data_t result = init;
		for(unsigned int grid = 0; grid < num_grids; ++ grid) {
			data_t temp_result;
			cudaStreamSynchronize(stream[grid]);
			cudaMemcpy(&temp_result, output[grid], sizeof(data_t), cudaMemcpyDeviceToHost);
			result = op(result, temp_result);
			output[grid] = base_output[grid];
			cudaStreamDestroy(stream[grid]);
			cudaFree(output[grid]);
		} // for
		//nvtxRangeEnd(nvtx2);

		return result;
	} // reduce()


	/* the original reduce - performs better than thrust on double, worse on single *
	 * it performs much better than thrust in RMC code */

	template <typename data_t, typename functor_t>
	__global__
	void reduce_block_multiple(const data_t* __restrict__ input, const unsigned int n,
						const data_t init, functor_t op,
						data_t* output, const unsigned int num_subtiles) {
		unsigned int output_i = blockIdx.x;
		unsigned int input_i = num_subtiles * blockDim.x * blockIdx.x + threadIdx.x;

		data_t *data_sh = (data_t*) d_data_sh;

		// start with init
		data_t sum = init;
		if(input_i < n) sum = input[input_i];

		// reduce the subtiles into single set (one sum for each thread)
		// this is the sequential component
		for(unsigned int i = 1; i < num_subtiles; ++ i) {
			input_i += blockDim.x;
			if(input_i < n) sum = op(sum, input[input_i]);
		} // for

		// latter half threads store their data to shared memory
		// first half reduce those to their registers
		// reduce in log_2(blockDim.x) steps
		unsigned int len = blockDim.x >> 1;
		while(len > 0) {
			if(threadIdx.x >= len && threadIdx.x < len << 1) data_sh[threadIdx.x] = sum;
			__syncthreads();
			if(threadIdx.x < len) sum = op(sum, data_sh[len + threadIdx.x]);
			len = len >> 1;
		} // while

		// write reduced output to global memory
		if(threadIdx.x == 0) output[output_i] = sum;
	} // reduce_block()


	template <typename iterator_t, typename data_t, typename functor_t>
	data_t reduce_multiple(iterator_t start, iterator_t end, data_t init, functor_t op) {

		unsigned int num_subtiles = NUM_SUBTILES_;
		unsigned int n = end - start;
		unsigned int block_size = BLOCK_DIM_REDUCE_;
		unsigned int n_grid_max = MAX_GRID_SIZE_ * block_size * num_subtiles;
		unsigned int num_grids = ceil((float) n / n_grid_max);

		//nvtxRangeId_t nvtx0 = nvtxRangeStart("orig_setup");
		data_t *output = NULL, *base_output = NULL;
		unsigned int output_size = MAX_GRID_SIZE_;
		cudaMalloc((void**) &output, output_size * sizeof(data_t));
		base_output = output;
		//nvtxRangeEnd(nvtx0);

		unsigned int d_shmem_size = block_size * sizeof(data_t);

		data_t result = init;
		for(unsigned int grid = 0; grid < num_grids; ++ grid) {	// for each grid
			// number of data elements to process in this grid:
			unsigned int n_grid = (grid == num_grids - 1) ?
									n - (num_grids - 1) * n_grid_max : n_grid_max;
			// the input data segment to process
			data_t *input = start + grid * n_grid_max;
			output = base_output;

			while(1) {
				unsigned int num_blocks = ceil((float) n_grid / (block_size * num_subtiles));
				//nvtxRangeId_t nvtx1 = nvtxRangeStart("orig_kernel");
				reduce_block_multiple <<< num_blocks, block_size, d_shmem_size >>>
								(input, n_grid, init, op, output, num_subtiles);
				cudaError_t err = cudaGetLastError();
				if(err != cudaSuccess) {
					std::cerr << "error: something went wrong in kernel launch: "
								<< cudaGetErrorString(err) << std::endl;
				} // if
				cudaThreadSynchronize();
				//nvtxRangeEnd(nvtx1);
				if(num_blocks == 1) break;
				n_grid = num_blocks;
				data_t *temp = input; input = output; output = temp;
			} // while

			//nvtxRangeId_t nvtx2 = nvtxRangeStart("orig_finalize");
			data_t temp_result;
			cudaMemcpy(&temp_result, &output[0], sizeof(data_t), cudaMemcpyDeviceToHost);
			result = op(result, temp_result);
			//nvtxRangeEnd(nvtx2);
		} // for

		output = base_output;
		cudaFree(output);

		return result;
	} // reduce()


	// another reduce: dont do multiple grids - performs comparable to above in double *
	// performs better than above in single, comparable to thrust */
	// use this as default :-)

	template <typename data_t, typename functor_t>
	__global__
	void reduce_block_single(const data_t* __restrict__ input, const unsigned int n,
						const data_t init, functor_t op,
						data_t* output, const unsigned int num_subtiles) {
		unsigned int th_output_i = blockIdx.x;
		unsigned int th_input_i = num_subtiles * blockDim.x * blockIdx.x + threadIdx.x;

		// start with init
		data_t sum = init;

		// reduce the grids and subtiles into single set (one sum for each thread)
		// this is the sequential component
		unsigned int grid_size = gridDim.x * blockDim.x * num_subtiles;
		unsigned int input_i = th_input_i;
		for(unsigned int grid_base_i = 0; grid_base_i < n; grid_base_i += grid_size) {
			input_i = grid_base_i + th_input_i;
			for(unsigned int i = 0; i < num_subtiles && input_i < n; ++ i, input_i += blockDim.x) {
				sum = op(sum, input[input_i]);
			} // for
		} // while

		// latter half threads store their data to shared memory
		// first half reduce those to their registers
		// reduce in log_2(blockDim.x) steps
		//data_t *data_sh = (data_t*) d_data_sh;
		//unsigned int len = blockDim.x >> 1;
		//while(len > 0) {
		//	if(threadIdx.x >= len && threadIdx.x < len << 1) data_sh[threadIdx.x] = sum;
		//	__syncthreads();
		//	if(threadIdx.x < len) sum = op(sum, data_sh[len + threadIdx.x]);
		//	len = len >> 1;
		//} // while
		data_t *data_sh = (data_t*) d_data_sh;
		for(unsigned int len = blockDim.x, len2 = blockDim.x >> 1; len2 > 0; len = len2, len2 = len >> 1) {
			if(threadIdx.x >= len2 && threadIdx.x < len) data_sh[threadIdx.x] = sum;
			__syncthreads();
			if(threadIdx.x < len2) sum = op(sum, data_sh[len2 + threadIdx.x]);
		} // while

		// write reduced output to global memory
		if(threadIdx.x == 0) output[th_output_i] = sum;
	} // reduce_block()


	template <typename iterator_t, typename data_t, typename functor_t>
	data_t reduce_single(iterator_t start, iterator_t end, data_t init, functor_t op) {

		unsigned int num_subtiles = NUM_SUBTILES_;
		unsigned int n = end - start;
		unsigned int block_size = BLOCK_DIM_REDUCE_ONE_;
		unsigned int n_grid_max = MAX_GRID_SIZE_ * block_size * num_subtiles;
		unsigned int num_grids = ceil((float) n / n_grid_max);

		//nvtxRangeId_t nvtx0 = nvtxRangeStart("one_setup");
		data_t *output = NULL, *base_output = NULL;
		data_t *input = NULL, *base_input = NULL;
		unsigned int output_size = MAX_GRID_SIZE_;
		unsigned int input_size = (n > MAX_GRID_SIZE_) ? n : MAX_GRID_SIZE_;
		cudaMalloc((void**) &output, output_size * sizeof(data_t));
		cudaMalloc((void**) &input, input_size * sizeof(data_t));
		// the input data segment to process
		cudaMemcpy(input, start, n * sizeof(data_t), cudaMemcpyDeviceToDevice);
		base_output = output;
		base_input = input;
		//nvtxRangeEnd(nvtx0);

		unsigned int d_shmem_size = block_size * sizeof(data_t);

		data_t result = init;
		unsigned int n_grid = min(n, n_grid_max);
		unsigned int n_i = n;
		output = base_output;

		while(1) {
			unsigned int num_blocks = ceil((double) n_grid / (block_size * num_subtiles));
			//nvtxRangeId_t nvtx1 = nvtxRangeStart("one_kernel");
			reduce_block_single <<< num_blocks, block_size, d_shmem_size >>>
							(input, n_i, init, op, output, num_subtiles);
			cudaError_t err = cudaGetLastError();
			if(err != cudaSuccess) {
				std::cerr << "error: something went wrong in kernel launch: "
							<< cudaGetErrorString(err) << std::endl;
			} // if
			cudaThreadSynchronize();
			//nvtxRangeEnd(nvtx1);
			if(num_blocks == 1) break;
			n_grid = num_blocks;
			n_i = num_blocks;
			data_t *temp = input; input = output; output = temp;
		} // while

		//nvtxRangeId_t nvtx2 = nvtxRangeStart("one_finalize");
		data_t temp_result;
		cudaMemcpy(&temp_result, &output[0], sizeof(data_t), cudaMemcpyDeviceToHost);
		result = op(result, temp_result);
		output = base_output;
		input = base_input;
		cudaFree(input);
		cudaFree(output);
		//nvtxRangeEnd(nvtx2);

		return result;
	} // reduce_single()


} // namespace cuda
} // namespace woo
