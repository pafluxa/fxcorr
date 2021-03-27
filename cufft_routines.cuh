// begin include guard
#ifndef CUFFTROUTINES_H
#define CUFFTROUTINES_H 
#pragma once

#include <stdio.h>
#include <cufft.h>

__device__ cufftComplex complex_conj(cufftComplex);

__device__ cufftComplex complex_scale(cufftComplex, float);

__device__ cufftComplex complex_mul(cufftComplex, cufftComplex);

__global__ void complex_pointwise_cms(cufftComplex*, const cufftComplex*, int, float);

__global__ void copy_halfforward_halfreverse(cufftComplex*, cufftComplex*, long);

// end include guard
#endif
