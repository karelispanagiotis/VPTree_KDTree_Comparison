# define the shell to bash
SHELL := /bin/bash

# define the CUDA/C++ compiler to use
CC = nvcc

# define includes
INCLUDES = -I../inc

# define flags
CFLAGS = -w -g
CUDAFLAGS = -arch=sm_35

test_sequential: 
	$(CC) $(INCLUDES) $(CFLAGS) tester.c ../src/vptree_sequential.c -o $@ -lm

test_cuda: 
	cd ..; make lib;
	$(CC) $(INCLUDES) $(CFLAGS) $(CUDAFLAGS) -x cu -dc tester.c
	$(CC) $(INCLUDES) $(CFLAGS) $(CUDAFLAGS) tester.o ../lib/vptree_cuda.o -o $@

test_cuda2:
	cd ..; make lib;
	$(CC) $(INCLUDES) $(CFLAGS) $(CUDAFLAGS) -x cu -dc tester_cuda.cpp
	$(CC) $(INCLUDES) $(CFLAGS) $(CUDAFLAGS) tester_cuda.o ../lib/vptree_cuda.o -o $@

test_knn:
	g++ $(INCLUDES) $(CFLAGS) tester_knn.c ../src/kNN.cpp ../src/vptree_sequential.c -o $@ -lm
clean: 
	rm test_* *.o