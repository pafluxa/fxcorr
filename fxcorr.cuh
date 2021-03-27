// begin include guard
#ifndef FXCORR_H
#define FXCORR_H

#include <cuda.h>
#include <cufft.h>
#include <stdlib.h>

struct fxcorrData_t 
{
    // number of samples in input signal
    long nsamples;
    // size in bytes of data arrays
    size_t sz; 
    // complex signals
    // signal[2*i] = real part of element i
    // signal[2*i+1] = imaginary part of element i
    float* signal1;
    float* signal2;
    float* out;
    // correlation output
    cufftComplex* output;
    // gpu device counterparts (cu_ prefix)
    cufftComplex* cu_signal1;
    cufftComplex* cu_signal2;
    cufftComplex* cu_output;
    // CUFFT handle (equivalent to FFTW plan)
    cufftHandle cufft_plan;
    // flag to indicate if cufft plan is available
    char hasCUFFTPlan;
    // flags to indicate if cuda buffers are allocated
    // initializes with -1 (undefined state)
    char hasCudaBuffers;
};
typedef struct fxcorrData_t fxcorrData;

extern "C" {
void fxcorr_attach_signals(
    float* s1, float* s2, long nsamples, fxcorrData* fxd
);
}

extern "C" {
void fxcorr_allocate_and_zeropad(
    fxcorrData* fxd
);
}

extern "C" {
void fxcorr_deallocate(
    fxcorrData* fxd
);
}

extern "C" {
int fxcorr_create_cufft_handle(
    fxcorrData* fxd
);
}

extern "C" {
int fxcorr_compute(
    fxcorrData* fxd
);
}

// end include guard
#endif
