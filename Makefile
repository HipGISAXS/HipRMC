##
# Project: HipRMC
#
# File: Makefile
# Created: January 27, 2013
# Modified: Jan 27, 2013
#
# Author: Abhinav Sarje <asarje@lbl.gov>
##

## base directories
#BOOST_DIR = /usr/local/boost_1_49_0
#MPI_DIR = /usr/local/openmpi-1.6
#CUDA_DIR = /usr/local/cuda-5.0
#HDF5_DIR = /usr/local/hdf5-1.8.9
#Z_DIR = /usr/local/zlib-1.2.7
#SZ_DIR = /usr/local/szip-2.1
#TIFF_LIB_DIR = /usr/local/lib
OPENCV_DIR = /usr/local/opencv
WOO_DIR = $(HOME)

## compilers
CXX = g++
#H5CC = $(HDF5_DIR)/bin/h5pcc
#NVCC = $(CUDA_DIR)/bin/nvcc

## compiler flags
CXX_FLAGS = -std=c++0x -Wall -Wextra #-lgsl -lgslcblas -lm
## gnu c++ compilers >= 4.3 support -std=c++0x [requirement for hipgisaxs 4.3.x <= g++ <= 4.6.x]
## gnu c++ compilers >= 4.7 also support -std=c++11, but they are not supported by cuda

## boost
#BOOST_INCL = -I $(BOOST_DIR)
#BOOST_LIBS = -L $(BOOST_DIR)/lib -lboost_system -lboost_filesystem -lboost_timer -lboost_chrono

## parallel hdf5
#HDF5_INCL = -I$(HDF5_DIR)/include -I$(SZ_DIR)/include -I$(Z_DIR)/include
#HDF5_LIBS = -L$(SZ_DIR)/lib -L$(Z_DIR)/lib -L$(HDF5_DIR)/lib -lhdf5 -lz -lsz -lm
#HDF5_FLAGS = -Wl,-rpath -Wl,$(HDF5_DIR)/lib
#HDF5_FLAGS += -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64 -D_POSIX_SOURCE -D_BSD_SOURCE

## mpi (openmpi)
#MPI_INCL = -I $(MPI_DIR)/include
#MPI_LIBS = -L $(MPI_DIR)/lib -lmpi_cxx -lmpi

## cuda
#CUDA_INCL = -I$(CUDA_DIR)/include
#CUDA_LIBS = -L$(CUDA_DIR)/lib64 -lcudart -lcufft
#NVCC_FLAGS = -Xcompiler -fPIC -Xcompiler -fopenmp -m 64
#NVCC_FLAGS += -gencode arch=compute_20,code=sm_20
#NVCC_FLAGS += -gencode=arch=compute_20,code=compute_20
#NVCC_FLAGS += -gencode arch=compute_20,code=sm_21
#NVCC_FLAGS += -gencode arch=compute_30,code=sm_30
#NVCC_FLAGS += -gencode arch=compute_35,code=sm_35
#NVCC_FLAGS += -Xptxas -v -Xcompiler -v -Xlinker -v --ptxas-options="-v"
#NVCC_FLAGS += -DGPUR -DFF_ANA_GPU #-G #-DFINDBLOCK #-DAXIS_ROT
#NVLIB_FLAGS = -Xlinker -lgomp
#NVLIB_FLAGS += -Wl,-rpath -Wl,$(CUDA_DIR)/lib64

## libtiff
#TIFF_LIBS = -L $(TIFF_LIB_DIR) -ltiff

## opencv
OPENCV_INCL = -I $(OPENCV_DIR)/include
OPENCV_LIBS = -L $(OPENCV_DIR)/lib -lopencv_core -lopencv_highgui

## woo
WOO_INCL = -I $(WOO_DIR)

## miscellaneous
MISC_INCL =
#MISC_FLAGS = -DKERNEL2 -DREDUCTION2 #-DAXIS_ROT
#MISC_FLAGS = -DFF_ANA_GPU

## choose optimization levels, debug flags, gprof flag, etc
OPT_FLAGS = -g -DDEBUG #-v #-pg
#OPT_FLAGS = -O3 -DNDEBUG #-v

## choose single or double precision here
PREC_FLAG =			# leave empty for single precision
#PREC_FLAG = -DDOUBLEP	# define this for double precision


## all includes
ALL_INCL = $(OPENCV_INCL) $(WOO_INCL)

## all libraries
ALL_LIBS = $(OPENCV_LIBS)


PREFIX = $(PWD)
BINARY_SIM = hiprmc
BIN_DIR = $(PREFIX)/bin
OBJ_DIR = $(PREFIX)/obj
SRC_DIR = $(PREFIX)/src

## all objects
OBJECTS_SIM = hiprmc.o

## the main binary
OBJ_BIN_SIM = $(patsubst %,$(OBJ_DIR)/%,$(OBJECTS_SIM))

$(BINARY_SIM): $(OBJ_BIN_SIM)
	$(CXX) -o $(BIN_DIR)/$@ $^ $(OPT_FLAGS) $(CXX_FLAGS) $(PREC_FLAG) $(MISC_FLAGS) $(ALL_LIBS)

## c++ compilation
_DEPS_CXX = %.hpp
DEPS_CXX = $(patsubst %,$(SRC_DIR)/%,$(_DEPS_CXX))

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(DEPS_CXX)
	$(CXX) -c $< -o $@ $(OPT_FLAGS) $(CXX_FLAGS) $(PREC_FLAG) $(ALL_INCL) $(MISC_FLAGS)

$(OBJ_DIR)/hiprmc.o: $(SRC_DIR)/hiprmc.cpp $(SRC_DIR)/rmc.hpp
	$(CXX) -c $< -o $@ $(OPT_FLAGS) $(CXX_FLAGS) $(PREC_FLAG) $(ALL_INCL) $(MISC_FLAGS)

all: hiprmc

.PHONY: clean

clean:
	rm -f $(OBJ_BIN_SIM) $(BIN_DIR)/$(BINARY_SIM)
