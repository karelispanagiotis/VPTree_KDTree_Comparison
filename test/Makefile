# define the shell to bash
SHELL := /bin/bash

# define the CUDA/C++ compiler to use
CC = nvcc

# define includes
INCLUDES = -I../inc

# define flags
CFLAGS = -w -g
CUDAFLAGS = -arch=sm_35 --fmad=false --ftz=false

# define kNN search dependencies
kNN_DEPENDS = ../src/vptree_sequential.cpp ../src/kNN.cpp ../src/kdtree_sequential.cpp

test_sequential: 
	$(CC) $(INCLUDES) $(CFLAGS) tester.cpp ../src/vptree_sequential.cpp -o $@ -lm

test_cuda: tester.cpp 
	cd ..; make vptree_cuda.o;
	$(CC) $(INCLUDES) $(CFLAGS) $(CUDAFLAGS) -x cu -dc tester.cpp
	$(CC) $(INCLUDES) $(CFLAGS) $(CUDAFLAGS) tester.o ../lib/vptree_cuda.o -o $@

test_knn: tester_knn.c $(kNN_DEPENDS)
	$(CC) -x cu $(INCLUDES) $(CFLAGS) $^  -o $@ -lm

test_mpi: tester_mpi.c ../src/knnring.cpp $(kNN_DEPENDS)
	mpic++ $(INCLUDES) $(CFLAGS) $^ -o $@

clean: 
	rm test_* *.o