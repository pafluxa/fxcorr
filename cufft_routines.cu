#include "cufft_routines.cuh"

// Complex conjugate
__device__ cufftComplex 
complex_conj(cufftComplex a)
{
    cufftComplex c;
    c.x =  a.x;
    c.y = -a.y;
    return c;
}

// Complex scale
__device__ cufftComplex 
complex_scale(cufftComplex a, float s)
{
    cufftComplex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}

// Complex multiplication
__device__ cufftComplex
complex_mul(cufftComplex a, cufftComplex b)
{
    cufftComplex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

// Complex pointwise conjugation/multiplication/scaling of arrays
// result is stored 
__global__ void 
complex_pointwise_cms(cufftComplex* a, const cufftComplex* b,
    int size, float scale)
{
    unsigned long i;
    cufftComplex atb;
    cufftComplex atbs;
    cufftComplex bconj;

    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    const int numThreads = blockDim.x * gridDim.x;
    
    for (i = threadID; i < size; i += numThreads)
    {
        bconj = complex_conj(b[i]);
        atb = complex_mul(a[i], bconj);
        atbs = complex_scale(atb, scale);
        a[i] = atbs;
    }
}

__global__ void 
copy_halfforward_halfreverse(cufftComplex* a, cufftComplex* out, 
    long nsamples)
{
    unsigned long i;
    unsigned long j;

    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    const int numThreads = blockDim.x * gridDim.x;
    
    for(i=threadID; i < nsamples; i += numThreads)
    {
        out[i] = out[i+nsamples];
    }
    for(i=threadID, j=nsamples*2-1; i<nsamples; i+=numThreads, j-=numThreads)
    {
        out[i+nsamples] = out[j];
    }
}
