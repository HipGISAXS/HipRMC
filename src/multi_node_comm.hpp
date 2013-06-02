/***
  *  Project:
  *
  *  File: multi_node_comm.hpp
  *  Created: Mar 18, 2013
  *  Modified: Mon 18 Mar 2013 03:03:20 PM PDT
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifdef __MULTI_NODE_COMM_HPP__
#define __MULTI_NODE_COMM_HPP__

#include <mpi.h>

namespace hir {

	class MultiNodeComm {

		public:
			MultiNodeComm(int narg, char** args, unsigned int n) {
				num_procs_ = n;
				MPI_Init(narg, args);
				all_world_ = MPI_COMM_WORLD;
				if(num_procs_ != all_world_.Get_size()) {
					std::cerr << "error: mismatch in number of MPI processes" << std::endl;
					exit(1);
				} // if
			} // MultiNodeComm()

			~MultiNodeComm() {
				MPI_Finalize();
			} // ~MultiNodeComm()

			int all_size() { return all_world_.Get_size(); }
			int rank() { return all_world_.Get_rank(); }

		private:
			unsigned int num_procs_;
			MPI_Comm all_world_;
	}; // class MultiNodeComm

} // namespace hir

#endif // __MULTI_NODE_COMM_HPP__
