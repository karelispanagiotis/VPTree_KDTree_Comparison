# ################################################
#
#					C Makefile
#		PDS Exercise 1: Vantage-Point Tree Build
#	  			Versions: Sequential, CUDA
#
#			Author: Panagiotis Karelis (9099)
# ################################################
#
# 'make' build executable file 'main'
# 'make lib' build the libraries .a
# 'make clean' removes all .o and executables
#

#define the shell to bash
SHELL := /bin/bash

#define the C/CUDA compiler to use
CC = nvcc

#define compile-time flags
CFLAGS = -w -O3 
CUDAFLAGS = -x cu -dc -arch=sm_35
#define directories containing header files
INCLUDES = -I./inc

#define Objects
OBJECTS = vptree_sequential.o vptree_cuda.o

########################################################################

lib: $(OBJECTS)
	
%.o : src/%.cpp
	$(CC) $(CFLAGS) $(CUDAFLAGS) $(INCLUDES) $< -o lib/$@

%.o : src/%.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o lib/$@

clean: 
	rm lib/*.o