/* Example showing the use of CUFFT for fast 1D-convolution using FFT. */

#include <cuda_correlation.h>

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cufft.h>

// Complex data type
typedef float2 Complex;
static __device__ inline Complex ComplexConj(Complex);
static __device__ inline Complex ComplexScale(Complex, float);
static __device__ inline Complex ComplexMul(Complex, Complex);
static __global__ void ComplexPointwiseConjMulAndScale(Complex*, const Complex*, int, float);

////////////////////////////////////////////////////////////////////////////////
void cudacorrelate_fft
(
    // input
    float re_signal1[], float im_signal1[], 
    float re_signal2[], float im_signal2[], int nsamples,
    // output
    float cs1s2[]
) {

    // Allocate host memory for signal 2. Allocate double the space 
    cufftComplex* signal1 = ( cufftComplex *)malloc(sizeof( cufftComplex ) * nsamples*2 );
    // Initalize the memory for the signal2
    for (unsigned int i = 0; i < nsamples; i++) {
        signal1[i].x = re_signal1[i];
        signal1[i].y = im_signal1[i];
    }
    // Zero pad
    for (unsigned int i = nsamples; i < 2*nsamples; i++) {
        signal1[i].x = 0.0;
        signal1[i].y = 0.0;
    }
    
    // Allocate host memory for signal 2. Allocate double the space 
    cufftComplex* signal2 = ( cufftComplex *)malloc(sizeof( cufftComplex ) * nsamples*2 );
    
    // Initalize the memory for the signal2
    for (unsigned int i = 0; i < nsamples; i++) {
        signal2[i].x = re_signal2[i];
        signal2[i].y = im_signal2[i];
    }
    // Zero pad
    for (unsigned int i = nsamples; i < 2*nsamples; i++) {
        signal2[i].x = 0.0;
        signal2[i].y = 0.0;
    }
    


    // Allocate device memory for signal1
    cufftComplex* d_signal1;
    cudaMalloc((void**)&d_signal1, sizeof(cufftComplex) * nsamples * 2 );
    // Copy host memory to device
    cudaMemcpy(d_signal1, signal1, sizeof(cufftComplex) * nsamples * 2, cudaMemcpyHostToDevice);
    
    // Allocate device memory for signal2
    cufftComplex* d_signal2;
    cudaMalloc((void**)&d_signal2, sizeof(cufftComplex) * nsamples * 2 );
    // Copy host memory to device
    cudaMemcpy(d_signal2, signal2, sizeof(cufftComplex) * nsamples * 2 , cudaMemcpyHostToDevice);

    // CUFFT plan
    cufftHandle plan; 
    if( cufftPlan1d(&plan, nsamples*2, CUFFT_C2C, 1) != CUFFT_SUCCESS )
    {
        fprintf( stderr, "CUDA FFT plan creation has failed.\n" );
        return;
    }

    // Transform signal and kernel
    if( cufftExecC2C(plan, d_signal1, d_signal1, CUFFT_FORWARD) != CUFFT_SUCCESS )
    {
        fprintf( stderr, "Launching C2C FFT has failed.\n" );
        return ;
    }
    
    if( cufftExecC2C(plan, d_signal2, d_signal2, CUFFT_FORWARD) != CUFFT_SUCCESS )
    {
        fprintf( stderr, "Launching C2C FFT has failed.\n" );
        return ;
    }

    if( cudaThreadSynchronize() != cudaSuccess )
	{
		fprintf( stderr, "CUDA error: failed to synchronize!.\n" ); 
	}
    
    // Multiply the coefficients together and normalize the result
    ComplexPointwiseConjMulAndScale<<<32, 256>>>(d_signal1, d_signal2, nsamples*2, 1.0f/(2*nsamples-1) );

    // Transform signal back
    //printf("Transforming signal back cufftExecC2C\n"); 
    if( cufftExecC2C(plan, d_signal1, d_signal1, CUFFT_INVERSE) != CUFFT_SUCCESS )
    {
        fprintf( stderr, "Launching C2C FFT has failed.\n" );
        return ;
    }
    
    // Copy fourier transform back to host
    cufftComplex* correlation = (cufftComplex*)malloc(sizeof(cufftComplex) * nsamples*2 );;
    cudaMemcpy( correlation, d_signal1, sizeof(cufftComplex)*nsamples*2, cudaMemcpyDeviceToHost);

    // Copy correlation to host. First half in ascending order
    for( int i=0; i < nsamples ; i++ )
    {    
        cs1s2[i] = correlation[nsamples+i].x/nsamples;
    }
    // Copy correlation to host. Second half in descending order
    for( int i=0, j=2*nsamples-1; i < nsamples ; i++,j-- )
    {    
        cs1s2[i+nsamples] = correlation[j].x/nsamples;
    }
    
    //Destroy CUFFT context
    cufftDestroy(plan);

    // cleanup memory
    free(signal1);
    free(signal2);
    cudaFree(d_signal1);
    cudaFree(d_signal2);

}

////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex conjugate
static __device__ inline Complex ComplexConj( Complex a )
{
    Complex c;
    c.x =  a.x;
    c.y = -a.y;
    return c;
}

// Complex scale
static __device__ inline Complex ComplexScale(Complex a, float s)
{
    Complex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}

// Complex multiplication
static __device__ inline Complex ComplexMul(Complex a, Complex b)
{
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

// Complex pointwise multiplication
static __global__ void ComplexPointwiseConjMulAndScale(Complex* a, const Complex* b, int size, float scale)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
        a[i] = ComplexScale(ComplexMul(a[i], ComplexConj(b[i])), scale);
}

