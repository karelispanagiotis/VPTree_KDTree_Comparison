# define the shell to bash
SHELL := /bin/bash

# define the CUDA/C++ compiler to use
CC = nvcc

# define includes
INCLUDES = -I../inc

# define flags
CFLAGS = -w -O3 -Xcompiler -fcilkplus
CUDAFLAGS = --fmad=false --ftz=false

# define kNN search dependencies
kNN_DEPENDS = ../src/kNN.cpp ../src/vptree_cuda.cu ../src/kdtree_cuda.cu

test_sequential: tester.cpp ../src/vptree_sequential.cpp ../src/utilities.cpp
	$(CC) $(INCLUDES) $(CFLAGS) $^ -o $@ -lm

test_cuda: tester.cpp 
	cd ..; make vptree_cuda.o;
	$(CC) $(INCLUDES) $(CFLAGS) $(CUDAFLAGS) -x cu -dc tester.cpp
	$(CC) $(INCLUDES) $(CFLAGS) $(CUDAFLAGS) tester.o ../lib/vptree_cuda.o -o $@

test_knn: tester_knn.cpp $(kNN_DEPENDS) ../src/utilities.cpp
	$(CC) $(INCLUDES) $(CFLAGS) $^  -o $@ -lm

test_mpi: tester_mpi.c ../src/knnring.cpp ../src/kNN.cpp ../src/vptree_sequential.cpp ../src/kdtree_sequential.cpp ../src/utilities.cpp
	mpic++ $(INCLUDES) $(CFLAGS) $^ -o $@

clean: 
	rm test_* *.o