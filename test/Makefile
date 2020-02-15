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
kNN_DEPENDS = tester_mpi.c ../src/vptree_sequential.cpp ../src/kNN.cpp ../src/knnring.cpp ../src/kdtree_sequential.cpp

test_sequential: 
	$(CC) $(INCLUDES) $(CFLAGS) tester.cpp ../src/vptree_sequential.cpp -o $@ -lm

test_cuda: 
	cd ..; make vptree_cuda.o;
	$(CC) $(INCLUDES) $(CFLAGS) $(CUDAFLAGS) -x cu -dc tester.c
	$(CC) $(INCLUDES) $(CFLAGS) $(CUDAFLAGS) tester.o ../lib/vptree_cuda.o -o $@

test_knn: tester_knn.c ../src/kdtree_sequential.cpp ../src/kNN.cpp
	g++ $(INCLUDES) $(CFLAGS) tester_knn.c ../src/kNN.cpp ../src/kdtree_sequential.cpp -o $@ -lm

test_mpi: $(kNN_DEPENDS)
	mpic++ $(INCLUDES) $(CFLAGS) $^ -o $@

clean: 
	rm test_* *.o