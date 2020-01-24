# define the shell to bash
SHELL := /bin/bash

# define the C/C++ compiler to use,default here is clang
CC = gcc-7

test_sequential:
	cd ../; make lib;
	cp ../lib/*.o ../inc/vptree.h ./;
	$(CC) tester.c vptree_sequential.o -o $@ -lm
	./test_sequential

clean: 
	rm *.h *.o test_*