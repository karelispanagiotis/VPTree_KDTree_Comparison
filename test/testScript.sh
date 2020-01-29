#!/bin/bash

make test_sequential;

echo {10,100,200,400,1000,5000}" "{2,4,5,8,53} | xargs -n 2 ./test_sequential

make clean;