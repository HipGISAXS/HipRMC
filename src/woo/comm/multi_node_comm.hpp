/***
  *  Project:
  *
  *  File: multi_node_comm.hpp
  *  Created: Mar 18, 2013
  *  Modified: Wed 16 Oct 2013 12:19:40 PM PDT
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef __MULTI_NODE_COMM_HPP__
#define __MULTI_NODE_COMM_HPP__

#ifdef USE_MPI

#include <mpi.h>
#include <complex>
#include <map>
#include <iostream>

namespace woo {

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

		protected:

			inline int size() const { int size; MPI_Comm_size(world_, &size); return size; }
			inline int rank() const { return rank_; }
			inline bool is_idle() const { return idle_; }
			inline bool is_valid() const { return valid_; }
			inline int master() const { return master_rank_; }
			inline bool is_master() const { return (master_rank_ == rank_); }

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
				return *this;
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
				return *this;
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

			inline bool free() {
				MPI_Comm_free(&world_);
				valid_ = false;
				return true;
			} // free()

			/*inline MultiNodeComm* dup() {
				MPI_Comm new_comm;
				MPI_Comm_dup(world_, &new_comm);
				MultiNodeComm *new_mncomm = new MultiNodeComm(new_comm);
				int rankie;
				MPI_Comm_rank(new_comm, &rankie);
				return new_mncomm;
			} // dup()*/

			inline bool broadcast(float* data, int size) const {
				MPI_Bcast(&(*data), size, MPI_FLOAT, master_rank_, world_);
				return true;
			} // broadcast()

			inline bool broadcast(double* data, int size) const {
				MPI_Bcast(&(*data), size, MPI_DOUBLE, master_rank_, world_);
				return true;
			} // broadcast()

			inline bool broadcast(int* data, int size) const {
				MPI_Bcast(&(*data), size, MPI_INT, master_rank_, world_);
				return true;
			} // broadcast()

			inline bool broadcast(unsigned int* data, int size) const {
				MPI_Bcast(&(*data), size, MPI_UNSIGNED, master_rank_, world_);
				return true;
			} // broadcast()

			inline bool broadcast(std::complex<float>* data, int size, int source) const {
				MPI_Bcast(&(*data), size, MPI_COMPLEX, source, world_);
				return true;
			} // broadcast()

			inline bool broadcast(std::complex<double>* data, int size, int source) const {
				MPI_Bcast(&(*data), size, MPI_DOUBLE_COMPLEX, source, world_);
				return true;
			} // broadcast()

			inline bool scan_sum(int in, int& out) const {
				if(MPI_Scan(&in, &out, 1, MPI_INT, MPI_SUM, world_) != MPI_SUCCESS)
					return false;
				return true;
			} // scan_sum()

			inline bool scan_sum(unsigned int in, unsigned int& out) const {
				if(MPI_Scan(&in, &out, 1, MPI_UNSIGNED, MPI_SUM, world_) != MPI_SUCCESS)
					return false;
				return true;
			} // scan_sum()

			inline bool gather(int* sbuf, int scount, int* rbuf, int rcount) const {
				MPI_Gather(sbuf, scount, MPI_INT, rbuf, rcount, MPI_INT, master_rank_, world_);
				return true;
			} // gather()

			inline bool gather(float* sbuf, int scount, float* rbuf, int rcount) const {
				MPI_Gather(sbuf, scount, MPI_FLOAT, rbuf, rcount, MPI_FLOAT, master_rank_, world_);
				return true;
			} // gather()

			inline bool gather(double* sbuf, int scount, double* rbuf, int rcount) const {
				MPI_Gather(sbuf, scount, MPI_DOUBLE, rbuf, rcount, MPI_DOUBLE, master_rank_, world_);
				return true;
			} // gather()

			inline bool allgather(int* sbuf, int scount, int* rbuf, int rcount) const {
				MPI_Allgather(sbuf, scount, MPI_INT, rbuf, rcount, MPI_INT, world_);
				return true;
			} // allgather()

			inline bool allgather(unsigned int* sbuf, int scount, unsigned int* rbuf, int rcount) const {
				MPI_Allgather(sbuf, scount, MPI_UNSIGNED, rbuf, rcount, MPI_UNSIGNED, world_);
				return true;
			} // allgather()

			inline bool gatherv(float* sbuf, int scount, float* rbuf, int* rcount, int* displs) const {
				MPI_Gatherv(sbuf, scount, MPI_FLOAT, rbuf, rcount, displs, MPI_FLOAT,
							master_rank_, world_);
				return true;
			} // gatherv()

			inline bool gatherv(double* sbuf, int scount, double* rbuf, int* rcount, int* displs) const {
				MPI_Gatherv(sbuf, scount, MPI_DOUBLE, rbuf, rcount, displs, MPI_DOUBLE,
							master_rank_, world_);
				return true;
			} // gatherv()

			inline bool gatherv(std::complex<float>* sbuf, int scount,
									std::complex<float>* rbuf, int* rcount, int* displs) const {
				MPI_Gatherv(sbuf, scount, MPI_COMPLEX, rbuf, rcount, displs, MPI_COMPLEX,
							master_rank_, world_);
				return true;
			} // gatherv()

			inline bool gatherv(std::complex<double>* sbuf, int scount,
									std::complex<double>* rbuf, int* rcount, int* displs) const {
				MPI_Gatherv(sbuf, scount, MPI_DOUBLE_COMPLEX,
								rbuf, rcount, displs, MPI_DOUBLE_COMPLEX, master_rank_, world_);
				return true;
			} // gatherv()

			inline bool scatterv(unsigned int* sbuf, int* scount, int* displs, unsigned int* rbuf,
									int rcount) const {
				MPI_Scatterv(sbuf, scount, displs, MPI_UNSIGNED,
								rbuf, rcount, MPI_UNSIGNED, master_rank_, world_);
				return true;
			} // scatterv()

			inline bool scatterv(float* sbuf, int* scount, int* displs, float* rbuf, int rcount) const {
				MPI_Scatterv(sbuf, scount, displs, MPI_FLOAT,
								rbuf, rcount, MPI_FLOAT, master_rank_, world_);
				return true;
			} // scatterv()

			inline bool scatterv(double* sbuf, int* scount, int* displs, double* rbuf, int rcount) const {
				MPI_Scatterv(sbuf, scount, displs, MPI_DOUBLE,
								rbuf, rcount, MPI_DOUBLE, master_rank_, world_);
				return true;
			} // scatterv()

			inline bool allreduce_sum(float in, float& out) const {
				MPI_Allreduce(&in, &out, 1, MPI_FLOAT, MPI_SUM, world_);
				return true;
			} // allreduce_sum()

			inline bool allreduce_sum(double in, double& out) const {
				MPI_Allreduce(&in, &out, 1, MPI_DOUBLE, MPI_SUM, world_);
				return true;
			} // allreduce_sum()

			inline bool barrier() const {
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


	typedef std::map <const std::string, MultiNodeComm> multi_node_comm_map_t;

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
			inline int size() const { return comms_.at("world").size(); }
			inline int rank() const { return comms_.at("world").rank(); }
			inline bool is_master() const { return (comms_.at("world").rank() == comms_.at("world").master()); }
			inline bool is_idle() const { return comms_.at("world").is_idle(); }
			inline int master() const { return comms_.at("world").master(); }

			inline int size(const std::string key) const { return comms_.at(key).size(); }
			inline int rank(const std::string key) const { return comms_.at(key).rank(); }
			inline bool is_master(const std::string key) const { return comms_.at(key).is_master(); }
			inline bool is_idle(const std::string key) const { return comms_.at(key).is_idle(); }
			inline int master(const std::string key) const { return comms_.at(key).master(); }

			inline void set_idle(const std::string key) { comms_.at(key).set_idle(); }

			// number of communicators including the world
			inline int num_comms() const { return comms_.size(); }

			/**
			 * Create new communicators
			 */

			bool split(const std::string new_key, const std::string key, int color) {
				comms_[new_key] = comms_.at(key).split(color);
				return true;
			} // split()

			bool dup(const std::string new_key, const std::string key) {
				comms_[new_key] = comms_.at(key).dup();
				return true;
			} // dup()

			bool free(const std::string key) {
				comms_.at(key).free();
				comms_.erase(key);
				return true;
			} // free

			/**
			 * Broadcasts
			 */

			bool broadcast(const std::string key, float* data, int size) const {
				return comms_.at(key).broadcast(data, size);
			} // send_broadcast()

			bool broadcast(const std::string key, double* data, int size) const {
				return comms_.at(key).broadcast(data, size);
			} // send_broadcast()

			bool broadcast(const std::string key, int* data, int size) const {
				return comms_.at(key).broadcast(data, size);
			} // send_broadcast()

			bool broadcast(const std::string key, unsigned int* data, int size) const {
				return comms_.at(key).broadcast(data, size);
			} // send_broadcast()

			bool broadcast(const std::string key, std::complex<float>* data, int size, int source) const {
				return comms_.at(key).broadcast(data, size, source);
			} // send_broadcast()

			bool broadcast(const std::string key, std::complex<double>* data, int size, int source) const {
				return comms_.at(key).broadcast(data, size, source);
			} // send_broadcast()

			/**
			 * Scans
			 */

			bool scan_sum(const std::string key, int in, int& out) const {
				return comms_.at(key).scan_sum(in, out);
			} // scan_sum()

			bool scan_sum(const std::string key, unsigned int in, unsigned int& out) const {
				return comms_.at(key).scan_sum(in, out);
			} // scan_sum()

			/**
			 * Gatherers
			 */

			bool allgather(const std::string key, int* sbuf, int scount, int* rbuf, int rcount) const {
				return comms_.at(key).allgather(sbuf, scount, rbuf, rcount);
			} // allgather()

			bool allgather(const std::string key, unsigned int* sbuf, int scount, unsigned int* rbuf, int rcount) const {
				return comms_.at(key).allgather(sbuf, scount, rbuf, rcount);
			} // allgather()

			inline bool gather(const std::string key, int* sbuf, int scount, int* rbuf, int rcount) const {
				return comms_.at(key).gather(sbuf, scount, rbuf, rcount);
			} // gather()

			inline bool gather(const std::string key, float* sbuf, int scount, float* rbuf, int rcount) const {
				return comms_.at(key).gather(sbuf, scount, rbuf, rcount);
			} // gather()

			inline bool gather(const std::string key, double* sbuf, int scount, double* rbuf, int rcount) const {
				return comms_.at(key).gather(sbuf, scount, rbuf, rcount);
			} // gather()

			inline bool gatherv(const std::string key, float* sbuf, int scount,
									float* rbuf, int* rcount, int* displs) const {
				return comms_.at(key).gatherv(sbuf, scount, rbuf, rcount, displs);
			} // gatherv()

			inline bool gatherv(const std::string key, double* sbuf, int scount,
									double* rbuf, int* rcount, int* displs) const {
				return comms_.at(key).gatherv(sbuf, scount, rbuf, rcount, displs);
			} // gatherv()

			inline bool gatherv(const std::string key, std::complex<float>* sbuf, int scount,
									std::complex<float>* rbuf, int* rcount, int* displs) const {
				return comms_.at(key).gatherv(sbuf, scount, rbuf, rcount, displs);
			} // gatherv()

			inline bool gatherv(const std::string key, std::complex<double>* sbuf, int scount,
									std::complex<double>* rbuf, int* rcount, int* displs) const {
				return comms_.at(key).gatherv(sbuf, scount, rbuf, rcount, displs);
			} // gatherv()

			/**
			 * Reducers
			 */

			inline bool allreduce_sum(const std::string key, float in, float& out) const {
				return comms_.at(key).allreduce_sum(in, out);
			} // allreduce_sum()

			inline bool allreduce_sum(const std::string key, double in, double& out) const {
				return comms_.at(key).allreduce_sum(in, out);
			} // allreduce_sum()

			/**
			 * Scatterers
			 */

			inline bool scatterv(const std::string key, unsigned int* sbuf, int* scount, int* displs,
									unsigned int* rbuf, int rcount) const {
				return comms_.at(key).scatterv(sbuf, scount, displs, rbuf, rcount);
			} // scatterv()

			inline bool scatterv(const std::string key, float* sbuf, int* scount, int* displs,
									float* rbuf, int rcount) const {
				return comms_.at(key).scatterv(sbuf, scount, displs, rbuf, rcount);
			} // scatterv()

			inline bool scatterv(const std::string key, double* sbuf, int* scount, int* displs,
									double* rbuf, int rcount) const {
				return comms_.at(key).scatterv(sbuf, scount, displs, rbuf, rcount);
			} // scatterv()

			/**
			 * Barriers
			 */

			bool barrier() const {
				return comms_.at("world").barrier();
			} // barrier()

			bool barrier(const std::string key) const {
				return comms_.at(key).barrier();
			} // barrier()

		private:
			unsigned int world_num_procs_;		// total number of processes
			multi_node_comm_map_t comms_;		// list of all communicators in the world

	}; // class MultiNode

} // namespace woo

#endif // USE_MPI

#endif // __MULTI_NODE_COMM_HPP__
