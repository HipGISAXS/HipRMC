/***
  *  Project:
  *
  *  File: multi_node_comm.hpp
  *  Created: Mar 18, 2013
  *  Modified: Fri 13 Sep 2013 09:16:38 AM PDT
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef __MULTI_NODE_COMM_HPP__
#define __MULTI_NODE_COMM_HPP__

#ifdef USE_MPI

#include <mpi.h>

namespace hir {

	static const int MASTER_RANK = 0;

	/**
	 * Communicator
	 */
	class MultiNodeComm {

		public:
			MultiNodeComm() :
				valid_(false), num_procs_(0), idle_(false), master_rank_(MASTER_RANK), rank_(0) {
				// create communicator ...
			} // MultiNodeComm()

			MultiNodeComm(const MPI_Comm& comm) {
				world_ = comm;
				num_procs_ = size();
				valid_ = true;
				idle_ = false;
				master_rank_ = MASTER_RANK;
				MPI_Comm_rank(world_, &rank_);
			} // operator=()

			~MultiNodeComm() {
				// destroy communicator ...
			} // ~MultiNodeComm()

			inline int size() const { int size; MPI_Comm_size(world_, &size); return size; }
			inline int rank() const { return rank_; }
			inline bool is_idle() const { return idle_; }
			inline bool is_valid() const { return valid_; }
			inline int master() const { return master_rank_; }
			inline int is_master() const { return (master_rank_ == rank_); }

			MultiNodeComm& operator=(const MPI_Comm& comm) {
				if(valid_) {
					std::cerr << "warning: assigning communicator to an already valid communicator. "
								<< "proceeding as is." << std::endl;
				} // if
				world_ = comm;
				num_procs_ = size();
				valid_ = true;
				idle_ = false;
				master_rank_ = MASTER_RANK;
				MPI_Comm_rank(world_, &rank_);
			} // operator=()

			MultiNodeComm& operator=(const MultiNodeComm& comm) {
				if(valid_) {
					std::cerr << "warning: assigning communicator to an already valid communicator. "
								<< "proceeding as is." << std::endl;
				} // if
				world_ = comm.world_;
				num_procs_ = size();
				valid_ = comm.valid_;
				idle_ = comm.idle_;
				master_rank_ = comm.master_rank_;
				rank_ = comm.rank_;
			} // operator=()

			inline void set_idle() { idle_ = true; }

			inline MultiNodeComm split(int color) {
				MPI_Comm new_comm;
				MPI_Comm_split(world_, color, rank_, &new_comm);
				MultiNodeComm new_mncomm(new_comm);
				return new_mncomm;
			} // split()

			inline MultiNodeComm dup() {
				MPI_Comm new_comm;
				MPI_Comm_dup(world_, &new_comm);
				MultiNodeComm new_mncomm(new_comm);
				int rankie;
				MPI_Comm_rank(new_comm, &rankie);
				return new_mncomm;
			} // dup()

			/*inline MultiNodeComm* dup() {
				MPI_Comm new_comm;
				MPI_Comm_dup(world_, &new_comm);
				MultiNodeComm *new_mncomm = new MultiNodeComm(new_comm);
				int rankie;
				MPI_Comm_rank(new_comm, &rankie);
				return new_mncomm;
			} // dup()*/

			inline bool broadcast(float* data, int size) {
				MPI_Bcast(&(*data), size, MPI_FLOAT, master_rank_, world_);
				return true;
			} // broadcast()

			inline bool broadcast(double* data, int size) {
				MPI_Bcast(&(*data), size, MPI_DOUBLE, master_rank_, world_);
				return true;
			} // broadcast()

			inline bool broadcast(unsigned int* data, int size) {
				MPI_Bcast(&(*data), size, MPI_UNSIGNED, master_rank_, world_);
				return true;
			} // broadcast()

			inline bool barrier() {
				MPI_Barrier(world_);
				return true;
			} // barrier()

		private:
			unsigned int num_procs_;	// number of processes in this communicator
			MPI_Comm world_;			// MPI communicator for this communicator
			bool valid_;
			bool idle_;					// status of a node
			int rank_;					// my rank
			int master_rank_;			// who is the master here

			friend class MultiNode;

	}; // class MultiNodeComm


	typedef std::map <const char*, MultiNodeComm> multi_node_comm_map_t;

	/**
	 * For communication - TODO: make it singleton
	 */
	class MultiNode {

		public:
			MultiNode(int narg, char** args) {
				MPI_Init(&narg, &args);
				comms_.clear();
				MultiNodeComm universe(MPI_COMM_WORLD);
				world_num_procs_ = universe.size();
				comms_["world"] = universe;
			} // MultiNodeComm()

			~MultiNode() {
				MPI_Finalize();
			} // ~MultiNodeComm()


			bool init() {
				return true;
			} // init()

			// number of procs in the world
			inline int size() { return comms_["world"].size(); }
			inline int rank() { return comms_["world"].rank(); }
			inline bool is_master() { return (comms_["world"].rank() == comms_["world"].master()); }
			inline bool is_idle() { return comms_["world"].is_idle(); }
			inline int master() { return comms_["world"].master(); }

			inline int size(const char* key) { return comms_[key].size(); }
			inline int rank(const char* key) { return comms_[key].rank(); }
			inline bool is_master(const char* key) { return comms_[key].is_master(); }
			inline bool is_idle(const char* key) { return comms_[key].is_idle(); }
			inline int master(const char* key) { return comms_[key].master(); }

			inline void set_idle(const char* key) { comms_[key].set_idle(); }

			// number of communicators including the world
			inline int num_comms() const { return comms_.size(); }

			bool split(const char* new_key, const char* key, int color) {
				comms_[new_key] = comms_[key].split(color);
				return true;
			} // split()

			bool dup(const char* new_key, const char* key) {
				comms_[new_key] = comms_[key].dup();
				return true;
			} // dup()

			bool broadcast(const char* key, float* data, int size) {
				return comms_[key].broadcast(data, size);
			} // send_broadcast()

			bool broadcast(const char* key, double* data, int size) {
				return comms_[key].broadcast(data, size);
			} // send_broadcast()

			bool broadcast(const char* key, unsigned int* data, int size) {
				return comms_[key].broadcast(data, size);
			} // send_broadcast()

			bool barrier(const char* key) {
				return comms_[key].barrier();
			} // barrier()

		private:
			unsigned int world_num_procs_;		// total number of processes
			multi_node_comm_map_t comms_;		// list of all communicators in the world

	}; // class MultiNode

} // namespace hir

#endif // USE_MPI

#endif // __MULTI_NODE_COMM_HPP__
