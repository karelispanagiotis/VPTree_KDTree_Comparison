# ############################################################
#
#							Makefile
#	PDS Exercise 4: VP-Tree & KD-Tree Construction and Search
#	  				Versions: Sequential, CUDA
#	
#			Author: Panagiotis Karelis (9099)
# ############################################################
#
# 'make lib' build the libraries .o
# 'make clean' removes all .o 
#

#define the shell to bash
SHELL := /bin/bash

#define the C/CUDA compiler to use
CC = nvcc

#define compile-time flags
CFLAGS = -w -O3 -Xcompiler -fcilkplus
CUDAFLAGS = -dc --fmad=false --ftz=false

#define directories containing header files
INCLUDES = -I./inc

#define Objects
OBJECTS = vptree_sequential.o kdtree_sequential.o kNN.o vptree_cuda.o kdtree_cuda.o 

########################################################################

lib: $(OBJECTS)
	
%.o : src/%.cpp
	$(CC) $(CFLAGS) $(CUDAFLAGS) $(INCLUDES) $< -o lib/$@

%.o : src/%.cu
	$(CC) $(CFLAGS) $(CUDAFLAGS) $(INCLUDES) $< -o lib/$@

clean: 
	rm lib/*.o