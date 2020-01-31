# define the shell to bash
SHELL := /bin/bash

# define the CUDA/C++ compiler to use
CC = nvcc

# define includes
INCLUDES = -I../inc

# define flags
CFLAGS = -w
CUDAFLAGS = -arch=sm_35

test_sequential: 
	$(CC) $(INCLUDES) $(CFLAGS) tester.c ../src/vptree_sequential.c -o $@ -lm

test_cuda: 
	cd ..; make lib;
	$(CC) $(INCLUDES) $(CFLAGS) $(CUDAFLAGS) -x cu -dc tester.c
	$(CC) $(INCLUDES) $(CFLAGS) $(CUDAFLAGS) tester.o ../lib/vptree_cuda.o -o $@

clean: 
	rm test_* *.o