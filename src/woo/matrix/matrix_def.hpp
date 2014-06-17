/***
  *  Project: Woo Matrix Library
  *
  *  File: matrix_def.hpp
  *  Created: Dec 03, 2012
  *  Modified: Wed 12 Jun 2013 11:59:33 AM PDT
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

//#include <vector>
//#include <cstring>
//#include <iostream>

#ifndef _MATRIX_DEF_HPP_
#define _MATRIX_DEF_HPP_

namespace woo {

	template <typename value_type> class ColumnIterator;
	template <typename value_type> class RowIterator;

	template <typename value_type>
	class Matrix {
		protected:
			value_type *mat_;
			unsigned int num_dims_;				// number of dimensions
			std::vector<unsigned int> dims_;	// values of dimensions
			unsigned int capacity_;

			inline unsigned int total_elements() {
				unsigned int tot_elems = 1;
				for(unsigned int i = 0; i < num_dims_; ++ i) tot_elems *= dims_[i];
				return tot_elems;
			} // total_elements()


		public:
			// ////
			// default constructor: matrix size not known
			// ////
			Matrix(): mat_(NULL), num_dims_(0) {
			} // Matrix()


			// ////
			// generic constructor
			// ////
			Matrix(unsigned int num_dims):
					mat_(NULL), num_dims_(num_dims), capacity_(0) {
			} // Matrix()


			// ////
			// destructor
			// ////
			~Matrix() {
				if(mat_ != NULL) delete[] mat_;
			} // ~Matrix()


			// ////
			// ////
			bool init(const std::vector<unsigned int>& dims) {
				if(dims.size() != num_dims_) {
					std::cerr << "error: number of dimensions does not match list of dimension values"
								<< std::endl;
					return false;
				} // if
				dims_.clear();
				unsigned int tot_elems = 1;
				for(unsigned int i = 0; i < num_dims_; ++ i) {
					dims_.push_back(dims[i]);
					tot_elems *= dims[i];
				} // for
				unsigned int size = 256;
				while(true) {
					if(tot_elems <= size) {
						tot_elems = size;
						break;
					} // if
					size *= 2;
				} // while
				return reserve(size);
			} // init()


			// ////
			// reserve memory for given number of elements
			// ////
			bool reserve(unsigned int size) {
				capacity_ = size;
				if(mat_ != NULL) delete[] mat_;
				mat_ = new (std::nothrow) value_type[size];
				if(mat_ == NULL) {
					std::cerr << "error: failed to reserve memory for the matrix" << std::endl;
					return false;
				} // if
				memset(mat_, 0, size * sizeof(value_type));
				return true;
			} // reserve()


			// ////
			// if data is given, populate the matrix with it
			// ////
			bool populate(value_type* data) {
				// just need to memcpy everything
				unsigned int tot_elems = 1;
				for(unsigned int i = 0; i < num_dims_; ++ i) tot_elems *= dims_[i];
				memcpy(mat_, data, tot_elems * sizeof(value_type));
				return true;
			} // populate()

			// ////
			// a few accessors
			// ////
			unsigned int dims() const { return num_dims_; }
			unsigned int dim_size(unsigned int i) const { return dims_[i]; }
			unsigned int capacity() const { return capacity_; }

			value_type*& data() { return mat_; }

	}; // class Matrix


	template <typename value_type>
	class Matrix2D : public Matrix <value_type> {
		public:

			typedef ColumnIterator<value_type> col_iterator;
			typedef RowIterator<value_type> row_iterator;

			friend class ColumnIterator<value_type>;
			friend class RowIterator<value_type>;

			class IndexType {
				public:
					IndexType(): num_(0), idx_(0) { }
					IndexType(int a, int b): num_(a), idx_(b) { }
					~IndexType() { }

					bool operator==(IndexType other) {
						return (other.num_ == num_ && other.idx_ == idx_);
					} // operator==()

					int num_;		// column/row number
					int idx_;		// index within the column/row
			}; // class IndexType

			typedef IndexType index_type;
			static const index_type end_index; // = index_type(-1, -1);
			static const index_type begin_index; // = index_type(0, 0);


			// ////
			// constructor: for empty matrix
			// ////
			Matrix2D(unsigned int rows, unsigned int cols):
					Matrix<value_type>(2) {
			//		end_index(-1, -1), begin_index(0, 0) {
				num_rows_ = rows;
				num_cols_ = cols;
				std::vector<unsigned int> dims;
				dims.push_back(num_rows_);
				dims.push_back(num_cols_);
				this->init(dims);
			} // Matrix2D()

			// ////
			// constructor: for prefilled matrix
			// ////
			Matrix2D(unsigned int rows, unsigned int cols, value_type* data):
					Matrix<value_type>(2),
			//		end_index(-1, -1), begin_index(0, 0),
					num_rows_(rows), num_cols_(cols) {
				std::vector<unsigned int> dims;
				dims.push_back(rows);
				dims.push_back(cols);
				this->init(dims);
				this->populate(data);
			} // Matrix2D()

			// ////
			// copy constructor
			// ////
			Matrix2D(const Matrix2D& mat):
					Matrix<value_type>(2) {
			//		end_index(-1, -1), begin_index(0, 0) {
				num_rows_ = mat.num_rows_;
				num_cols_ = mat.num_cols_;
				this->num_dims_ = mat.num_dims_;
				std::vector<unsigned int> dims;
				dims.push_back(num_rows_);
				dims.push_back(num_cols_);
				this->init(dims);
				this->populate(mat.mat_);
			} // Matrix2D()


			// ////
			// assignment operator
			// ////
			Matrix2D& operator=(const Matrix2D& mat) {
				num_rows_ = mat.num_rows_;
				num_cols_ = mat.num_cols_;
				this->num_dims_ = mat.num_dims_;
				std::vector<unsigned int> dims;
				dims.push_back(num_rows_);
				dims.push_back(num_cols_);
				this->init(dims);
				this->populate(mat.mat_);
				return *this;
			} // Matrix2D()


			// ////
			// destructor
			// ////
			~Matrix2D() {
			} // ~Matrix2D()


			// ////
			// iterators
			// ////

			// ////
			// iterator to column
			// ////
			col_iterator column(unsigned int i) {
				if(i <= 0) {
					col_iterator start_col(0, num_rows_, num_cols_, this);
					return start_col;
				} // if
				if(i >= num_cols_) {
					col_iterator end_col(num_cols_ - 1, num_rows_, num_cols_, this);
					return end_col;
				} // if
				col_iterator col(i, num_rows_, num_cols_, this);
				return col;
			} // column()

			// ////
			// iterator to first column
			// ////
			col_iterator begin_col() {
				col_iterator start_col(0, num_rows_, num_cols_, this);
				return start_col;
			} // begin()

			// ////
			// iterator to column after last (end_index)
			// ////
			col_iterator end_col() {
				col_iterator last_col(num_cols_, num_rows_, num_cols_, this);
				return last_col;
			} // end()


			// ////
			// iterator to row
			// ////
			row_iterator row(unsigned int i) {
				if(i <= 0) {
					row_iterator start_row(0, num_cols_, num_rows_, this);
					return start_row;
				} // if
				if(i >= num_rows_) {
					row_iterator last_row(num_rows_ - 1, num_cols_, num_rows_, this);
					return last_row;
				} // if
				row_iterator row(i, num_cols_, num_cols_, this);
				return row;
			} // column()

			// ////
			// iterator to first row
			// ////
			row_iterator begin_row() {
				row_iterator start_row(0, num_cols_, num_rows_, this);
				return start_row;
			} // begin()

			// ////
			// iterator to row after last (end_index)
			// ////
			row_iterator end_row() {
				row_iterator last_row(num_rows_, num_cols_, num_rows_, this);
				return last_row;
			} // end()


			// ////
			// accessors
			// ////

			unsigned int num_cols() const { return num_cols_; }
			unsigned int num_rows() const { return num_rows_; }
			unsigned int size() const { return num_cols_ * num_rows_; }

			// ////
			// access (i, j)-th element
			// ////
			value_type& operator()(unsigned int i, unsigned int j) const {
				return this->mat_[num_cols_ * i + j];
			} // operator()()

			// ////
			// access an element through sequential indexing
			// ////
			value_type& operator[](unsigned int index) const {
				return this->mat_[index];
			} // operator[]()


			// ////
			// modifiers
			// ////

			// ////
			// fill matrix with a value
			// ////
			bool fill(value_type val) {
				#pragma omp parallel for collapse(2) shared(val)
				for(unsigned int i = 0; i < num_rows_; ++ i) {
					for(unsigned int j = 0; j < num_cols_; ++ j) {
						this->mat_[num_cols_ * i + j] = val;
					} // for
				} // for
				return true;
			} // fill()

			/*// specializer for uint
			bool fill(unsigned int val) {
				if(val == 0) {
					memset(this->mat_, 0, num_rows_ * num_cols_ * sizeof(unsigned int));
				} else {
					#pragma omp parallel for collapse(2) shared(val)
					for(unsigned int i = 0; i < num_rows_; ++ i) {
						for(unsigned int j = 0; j < num_cols_; ++ j) {
							this->mat_[num_cols_ * i + j] = val;
						} // for
					} // for
				} // if-else
				return true;
			} // fill()

			// specializer for int
			bool fill(int val) {
				if(val == 0) {
					memset(this->mat_, 0, num_rows_ * num_cols_ * sizeof(int));
				} else {
					#pragma omp parallel for collapse(2) shared(val)
					for(unsigned int i = 0; i < num_rows_; ++ i) {
						for(unsigned int j = 0; j < num_cols_; ++ j) {
							this->mat_[num_cols_ * i + j] = val;
						} // for
					} // for
				} // if-else
				return true;
			} // fill()*/

			// ////
			// insert stuff
			// ////

			// ////
			// insert a new row
			// ////
			//bool insert_row(unsigned int i, const std::vector<value_type>& row) {
			bool insert_row(unsigned int i, value_type* row, unsigned int size) {
				if(size != num_cols_) {
					std::cerr << "error: mismatching row size during insertion ("
								<< size << " != " << num_cols_ << ")" << std::endl;
					return false;
				} // if
				if(i > num_rows_ + 1) {
					std::cerr << "error: position is greater than resulting number of rows" << std::endl;
					return false;
				} // if
				unsigned int tot_elems = this->total_elements();
				value_type* temp = NULL;
				if(tot_elems + num_cols_ > this->capacity_) this->capacity_ *= 2;
				// TODO: mem allocation can be easily avoided here when there is enough capacity
				temp = new (std::nothrow) value_type[this->capacity_];
				if(temp == NULL) {
					std::cerr << "error: failed to resize memory during row insertion" << std::endl;
					return false;
				} // if
				// copy all memory until row i
				// copy new row
				// copy remaining rows
				memcpy(temp, this->mat_, i * num_cols_ * sizeof(value_type));
				memcpy(temp + i * num_cols_, row, num_cols_ * sizeof(value_type));
				memcpy(temp + (i + 1) * num_cols_, this->mat_ + i * num_cols_,
						(num_rows_ - i) * num_cols_ * sizeof(value_type));
				delete[] this->mat_;
				this->mat_ = temp;
				temp = NULL;
				++ num_rows_;
				++ this->dims_[0];

				return true;
			} // insert_row()


			// ////
			// insert a new column
			// ////
			//bool insert_col(unsigned int i, const std::vector<value_type>& col) {
			bool insert_col(unsigned int i, value_type* &col, unsigned int size) {
				if(size != num_rows_) {
					std::cerr << "error: mismatching column size during insertion ("
								<< size << " != " << num_rows_ << ")" << std::endl;
					return false;
				} // if
				if(i > num_cols_ + 1) {
					std::cerr << "error: position is greater than resulting number of columns" << std::endl;
					return false;
				} // if
				unsigned int tot_elems = this->total_elements();
				value_type* temp = NULL;
				if(tot_elems + num_rows_ > this->capacity_) this->capacity_ *= 2;
				// TODO: can mem allocation be avoided here when there is enough capacity?
				temp = new (std::nothrow) value_type[this->capacity_];
				if(temp == NULL) {
					std::cerr << "error: failed to resize memory during column insertion" << std::endl;
					return false;
				} // if
				// repeat for each row:
				// copy memory until column i
				// copy column i value
				// copy remaining cols
				for(unsigned int row = 0; row < num_rows_; ++ row) {
					memcpy(temp + row * (num_cols_ + 1), this->mat_ + row * num_cols_,
							i * sizeof(value_type));
					memcpy(temp + row * (num_cols_ + 1) + i, col + row, sizeof(value_type));
					memcpy(temp + row * (num_cols_ + 1) + i + 1, this->mat_ + row * num_cols_ + i,
							(num_cols_ - i) * sizeof(value_type));
				} // for
				delete[] this->mat_;
				this->mat_ = temp;
				temp = NULL;
				++ num_cols_;
				++ this->dims_[1];

				return true;
			} // insert_col()


			// ////
			// increase matrix size
			// ////

			// ////
			// increase rows - inserts num rows at the end
			// preserves initial data, initializes new rows to zero
			// ////
			bool incr_rows(unsigned int num) {
				num_rows_ += num;
				this->dims_[0] += num;
				unsigned int tot_elems = this->total_elements();
				if(tot_elems > this->capacity_) {
					this->capacity_ *= 2;
					value_type* temp = new (std::nothrow) value_type[this->capacity_];
					if(temp == NULL) {
						std::cerr << "error: failed to resize memory during row insertion" << std::endl;
						return false;
					} // if
					// copy all memory until row the original end
					memcpy(temp, this->mat_, (num_rows_ - num) * num_cols_ * sizeof(value_type));
					memset(temp + (num_rows_ - num) * num_cols_, 0, num * num_cols_ * sizeof(value_type));
					delete[] this->mat_;
					this->mat_ = temp;
					temp = NULL;
				} else {
					memset(this->mat_ + (num_rows_ - num) * num_cols_, 0,
							num * num_cols_ * sizeof(value_type));
				} // if-else
				return true;
			} // incr_rows()

			// ////
			// increase columns - inserts num elements at the end of each row (num cols)
			// preserves initial data, initializes new cols to zero
			// ////
			bool incr_columns(unsigned int num) {
				num_cols_ += num;
				this->dims_[1] += num;
				unsigned int tot_elems = this->total_elements();
				if(tot_elems > this->capacity_) this->capacity_ *= 2;
				// TODO: avoid mem allocation when possible
				value_type* temp = new (std::nothrow) value_type[this->capacity_];
				if(temp == NULL) {
					std::cerr << "error: failed to resize memory during column insertion" << std::endl;
					return false;
				} // if
				// repeat for each row:
				// copy memory until original last column
				// copy column i value
				// copy remaining cols
				for(unsigned int row = 0; row < num_rows_; ++ row) {
					memcpy(temp + row * num_cols_, this->mat_ + row * (num_cols_ - num),
							(num_cols_ - num) * sizeof(value_type));
					memset(temp + row * num_cols_ + num_cols_ - num, 0, num * sizeof(value_type));
				} // for
				delete[] this->mat_;
				this->mat_ = temp;
				temp = NULL;
				return true;
			} // incr_columns()


			// ////
			// resize the matrix to the new dimensions, and initializes to zero
			// does NOT preserve any initial data
			// ////
			bool resize(unsigned int new_rows, unsigned int new_cols) {
				num_rows_ = new_rows;
				num_cols_ = new_cols;
				this->num_dims_ = 2;
				std::vector<unsigned int> dims;
				dims.push_back(num_rows_);
				dims.push_back(num_cols_);
				this->init(dims);
				memset(this->mat_, 0, new_rows * new_cols * sizeof(value_type));
				return true;
			} // resize()


		private:
			unsigned int num_cols_;		// number of columns = row size
			unsigned int num_rows_;		// number of rows = col size

	}; // class Matrix2D

	
	template <typename value_type>
	const typename Matrix2D<value_type>::index_type Matrix2D<value_type>::begin_index(0, 0);
	template <typename value_type>
	const typename Matrix2D<value_type>::index_type Matrix2D<value_type>::end_index(-1, -1);

	// ////
	// begin and end index_type constants
	// ////
/*	template<> const Matrix2D<float>::index_type Matrix2D<float>::begin_index(0, 0);
	template<> const Matrix2D<float>::index_type Matrix2D<float>::end_index(-1, -1);
	template<> const Matrix2D<double>::index_type Matrix2D<double>::begin_index(0, 0);
	template<> const Matrix2D<double>::index_type Matrix2D<double>::end_index(-1, -1);
	template<> const Matrix2D<int>::index_type Matrix2D<int>::begin_index(0, 0);
	template<> const Matrix2D<int>::index_type Matrix2D<int>::end_index(-1, -1);
	template<> const Matrix2D<unsigned int>::index_type Matrix2D<unsigned int>::begin_index(0, 0);
	template<> const Matrix2D<unsigned int>::index_type Matrix2D<unsigned int>::end_index(-1, -1);
	template<> const Matrix2D<long int>::index_type Matrix2D<long int>::begin_index(0, 0);
	template<> const Matrix2D<long int>::index_type Matrix2D<long int>::end_index(-1, -1);
	template<> const Matrix2D<unsigned long int>::index_type Matrix2D<unsigned long int>::begin_index(0, 0);
	template<> const Matrix2D<unsigned long int>::index_type Matrix2D<unsigned long int>::end_index(-1, -1);
	template<> const Matrix2D<std::complex<float> >::index_type Matrix2D<std::complex<float> >::begin_index(0, 0);
	template<> const Matrix2D<std::complex<float> >::index_type Matrix2D<std::complex<float> >::end_index(-1, -1);
	template<> const Matrix2D<std::complex<double> >::index_type Matrix2D<std::complex<double> >::begin_index(0, 0);
	template<> const Matrix2D<std::complex<double> >::index_type Matrix2D<std::complex<double> >::end_index(-1, -1);
*/
	// ////
	// other functions involving above defined matrices
	// ////
	// ////
	// matrix addition: c = a + b
	// ////
	template <typename value_type>
	static bool matrix_add(Matrix2D<value_type>& a, Matrix2D<value_type>& b, Matrix2D<value_type>& c) {
		unsigned int nrows = a.num_rows();
		unsigned int ncols = a.num_cols();
		if(nrows != b.num_rows() || nrows != c.num_rows() ||
				ncols != b.num_cols() || ncols != c.num_cols()) {
			std::cerr << "error: matrices should have equal dimensions" << std::endl;
			return false;
		} // if

		value_type *a_mat, *b_mat, *c_mat;
		a_mat = a.data(); b_mat = b.data(); c_mat = c.data();
		#pragma omp parallel for
		for(unsigned int i = 0; i < nrows * ncols; ++ i) c_mat[i] = a_mat[i] + b_mat[i];
		return true;
	} // matrix_add()


	template <typename value_type>
	static bool matrix_min_max(const Matrix2D<value_type>& mat, value_type& min_val, value_type& max_val) {
		unsigned int nrows = mat.num_rows();
		unsigned int ncols = mat.num_cols();
		min_val = max_val = mat[0];
		for(unsigned int i = 0; i < nrows * ncols; ++ i) {
			if(min_val > mat[i]) min_val = mat[i];
			if(max_val < mat[i]) max_val = mat[i];
		} // for
		return true;
	} // matrix_min_max()

} // namespace woo

#endif // _MATRIX_DEF_HPP_
