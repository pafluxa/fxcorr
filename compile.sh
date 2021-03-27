#!/bin/bash
nvcc -c fxcorr.cu cufft_routines.cu
g++ -c main.c -I /usr/local/cuda/include
g++ -o test.exe main.o fxcorr.o cufft_routines.o -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -lcudart -lcufft
