# define the shell to bash
SHELL := /bin/bash

# define the C/C++ compiler to use
CC = gcc-7

# define includes
INCLUDES = -I../inc

# define flags
CFLAGS = -Wall -g

test_sequential: 
	$(CC) $(INCLUDES) $(CFLAGS) tester.c ../src/vptree_sequential.c -o $@ -lm


clean: 
	rm test_*