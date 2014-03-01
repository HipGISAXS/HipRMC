/***
  *  Project: Woo Matrix Library
  *
  *  File: matrix.hpp
  *  Created: Dec 03, 2012
  *  Modified: Sun 03 Feb 2013 02:15:27 PM PST
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

//#include <vector>
//#include <cstring>
//#include <iostream>

//#include "matrix.hpp"
#include "matrix_def.hpp"

#ifndef _ITERATORS_HPP_
#define _ITERATORS_HPP_

namespace woo {

	/* a generic iterator */
	template <typename value_type>
	class DimensionIterator {
		public:

			DimensionIterator(unsigned int d, unsigned int size, unsigned int i, value_type* mat):
				dim_num_(d), dim_size_(size), dim_index_(i), dim_pointer_(mat) { }
			~DimensionIterator() { }

			virtual void operator++() { }
			virtual void operator--() { }

		protected:

			unsigned int dim_num_;			// dimension number
			unsigned int dim_size_;			// size of the dimension
			unsigned int dim_index_;		// index of the current dimension vector
			value_type* dim_pointer_;		// pointer to current dimension vector

	}; // class DimensionIterator


	/* column vector for Matrix2D */
	template <typename value_type>
	class ColumnIterator : public DimensionIterator <value_type> {
		private:
			Matrix2D<value_type>* parent_mat_;					// is iterator of this object
			typename Matrix2D<value_type>::index_type index_;	// index in matrix

		public:
			/* create an iterator pointing to column 0 */
			ColumnIterator():
				DimensionIterator<value_type>(1, 0, 0, NULL) {
				index_.num_ = 0;
				index_.idx_ = 0;
			} // ColumnIterator()

			ColumnIterator(unsigned int i, unsigned int num_rows, unsigned int num_cols,
							Matrix2D<value_type>* mat): 
				DimensionIterator<value_type>(1, num_rows, i, mat->mat_) {
				parent_mat_ = mat;
				if(i >= num_cols) {
					index_ = Matrix2D<value_type>::end_index;
				} else {
					index_.num_ = i;
					index_.idx_ = 0;
				} // if-else
			} // ColumnIterator()

			/* increment to next column */
			void operator++() {		// next column
				if(index_ == Matrix2D<value_type>::end_index) {
					// do nothing
				} else if(index_.num_ == parent_mat_->num_cols_ - 1) {
					index_ = Matrix2D<value_type>::end_index;
					this->dim_pointer_ = NULL;
				} else {
					++ index_.num_;
					index_.idx_ = 0;
					unsigned int index = index_.idx_ * parent_mat_->num_cols_ + index_.num_;
					this->dim_pointer_ = &(parent_mat_->mat_[index]);
				} // if-else
			} // operator++()

			/* decrement to previous column */
			void operator--() {		// previous column
				if(index_ == Matrix2D<value_type>::end_index) {
					index_.num_ = parent_mat_->num_cols_ - 1;
				} else if(index_.num_ == 0) {
					index_ = Matrix2D<value_type>::begin_index;
				} else if(index_.num_ != 0) {
					-- index_.num_;
				} // if-else
				index_.idx_ = 0;
				unsigned int index = index_.idx_ * parent_mat_->num_cols_ + index_.num_;
				this->dim_pointer_ = &(parent_mat_->mat_[index]);
			} // operator--()

			/* assign a pointer to the iterator */
			void operator=(value_type* pointer) {
				this->dim_pointer_ = pointer;
				// what about index_ ??? ...
			} // operator=()

			/* return the i-th element of current column */
			value_type& operator[](unsigned int i) {
				if(i >= parent_mat_->num_rows_) {
					// return the last element
					return parent_mat_->mat_[(parent_mat_->num_rows_ - 1) * parent_mat_->num_cols_ +
												parent_mat_->num_rows_ - 1];
				} // if
			//	index_.idx_ = i;
				return parent_mat_->mat_[i * parent_mat_->num_cols_ + index_.num_];
			} // operator[]()

			/* comparison operators */

			bool operator==(ColumnIterator other) const {
				return (other.index_ == index_ && other.parent_mat_ == parent_mat_);
			} // operator==()

			bool operator!=(ColumnIterator other) const {
				return !(other.index_ == index_ && other.parent_mat_ == parent_mat_);
			} // operator!=()

			/* return the value of current column's current index */
			value_type value() const {
				unsigned int index = index_.idx_ * parent_mat_->num_cols_ + index_.num_;
				return parent_mat_->mat_[index];
			} // value()

			unsigned int size() const { return parent_mat_->num_rows_; }

	}; // class ColumnIterator


	/* row iterator for Matrix2D */
	template <typename value_type>
	class RowIterator : public DimensionIterator <value_type> {
		private:
			Matrix2D<value_type>* parent_mat_;					// is iterator of this object
			typename Matrix2D<value_type>::index_type index_;	// index in matrix

		public:
			/* create an iterator pointing to row 0 */
			RowIterator(): 
				DimensionIterator<value_type>(0, 0, 0, NULL) {
				index_.num_ = 0;
				index_.idx_ = 0;
			} // ColumnIterator()

			RowIterator(unsigned int i, unsigned int num_cols, unsigned int num_rows,
							Matrix2D<value_type>* mat): 
				DimensionIterator<value_type>(1, num_cols, i, mat->mat_) {
				parent_mat_ = mat;
				if(i >= num_rows) {
					index_ = Matrix2D<value_type>::end_index;
				} else {
					index_.num_ = i;
					index_.idx_ = 0;
				} // if-else
			} // ColumnIterator()

			/* increment to next row */
			void operator++() {
				if(index_ == Matrix2D<value_type>::end_index) {
					// do nothing
				} else if(index_.num_ == parent_mat_->num_rows_ - 1) {
					index_ = Matrix2D<value_type>::end_index;
					this->dim_pointer_ = NULL;
				} else {
					++ index_.num_;
					index_.idx_ = 0;
					unsigned int index = parent_mat_->num_cols_ * index_.num_ + index_.idx_;
					this->dim_pointer_ = &(parent_mat_->mat_[index]);
				} // if-else
			} // operator++()

			/* decrement to previous row */
			void operator--() {		// previous column
				if(index_ == Matrix2D<value_type>::end_index) {
					index_.num_ = parent_mat_->num_rows_ - 1;
				} else if(index_.num_ == 0) {
					index_ = Matrix2D<value_type>::begin_index;
				} else if(index_.num_ != 0) {
					-- index_.num_;
				} // if-else
				index_.idx_ = 0;
				unsigned int index = parent_mat_->num_cols_ * index_.num_ + index_.idx_;
				this->dim_pointer_ = &(parent_mat_->mat_[index]);
			} // operator--()

			/* assign a pointer to the iterator */
			void operator=(value_type* pointer) {
				this->dim_pointer_ = pointer;
				// what about index_ ?
			} // operator=()

			/* return the i-th element of current row */
			value_type& operator[](unsigned int i) {
				if(i >= parent_mat_->num_cols_) {
					return parent_mat_->mat_[(parent_mat_->num_rows_ - 1) * parent_mat_->num_cols_ +
												parent_mat_->num_cols_ - 1];
				} // if
			//	index_.idx_ = i;
				return parent_mat_->mat_[index_.num_ * parent_mat_->num_cols_ + i];
			} // operator[]()

			/* comparison operators */

			bool operator==(RowIterator other) const {
				return (other.index_ == index_ && other.parent_mat_ == parent_mat_);
			} // operator==()

			bool operator!=(RowIterator other) const {
				return !(other.index_ == index_ && other.parent_mat_ == parent_mat_);
			} // operator!=()

			/* return the value of current row's current index */
			value_type value() const {
				unsigned int index = index_.idx_ + parent_mat_->num_cols_ * index_.num_;
				return parent_mat_->mat_[index];
			} // value()

			unsigned int size() const { return parent_mat_->num_cols_; }

	}; // class RowIterator

} // namespace woo

#endif // _ITERATORS_HPP_
