#include <stdio.h>
#include "fxcorr.cuh"
#include "cufft_routines.cuh"

void _cudaCheck(cudaError_t err, const char *file, int line) {
   if (err != cudaSuccess) 
   {
       fprintf(stderr, "cuda error: %s at %s line %d", cudaGetErrorString(err), file, line);
       exit(-1);
   }
}
#define cudaCheck(ans) { _cudaCheck((ans), __FILE__, __LINE__); }

extern void 
fxcorr_attach_signals(float* s1, float* s2, long nsamples,
    fxcorrData* fxd)
{
    fxd->nsamples = nsamples;
    fxd->signal1 = s1;
    fxd->signal2 = s2; 
    fxd->out = (float*)malloc(2*sizeof(float)*nsamples);
};

extern void
fxcorr_allocate_and_zeropad(fxcorrData* fxd) // TODO: add CUDA checks
{
    // two is there because of zero-padding
    fxd->sz = (2*fxd->nsamples)*sizeof(cufftComplex);
    // allocate memory on device
    cudaCheck(cudaMalloc((void**)&fxd->cu_signal1, fxd->sz));
    cudaCheck(cudaMalloc((void**)&fxd->cu_signal2, fxd->sz));
    // memset is the fastest way to zero-pad
    cudaCheck(cudaMemset(fxd->cu_signal1, 0, fxd->sz));
    cudaCheck(cudaMemset(fxd->cu_signal2, 0, fxd->sz));
    // flag that buffers are good
    fxd->hasCudaBuffers = 1;
}

extern void 
fxcorr_deallocate(fxcorrData* fxd)
{
    //Destroy CUFFT context
    cufftDestroy(fxd->cufft_plan);

    // cleanup memory
    cudaFree(fxd->cu_signal1);
    cudaFree(fxd->cu_signal2);
    
    free(fxd->out);
}

extern int
fxcorr_create_cufft_handle(fxcorrData* fxd)
{
    if(cufftPlan1d(&(fxd->cufft_plan), 
        fxd->nsamples*2, CUFFT_C2C, 1) != CUFFT_SUCCESS )
    {
        fprintf(stderr, "CUDA FFT plan creation has failed.\n");
        return -1;
    }
    fxd->hasCUFFTPlan = 1;
    
    return 0;
}

extern int
fxcorr_compute(fxcorrData* fxd)
{
    // return -1 if no buffers are present
    if(fxd->hasCudaBuffers != 1)
    { 
        return -1;
    }
    
    // copy host data to GPU
    cudaMemcpy(fxd->cu_signal1, fxd->signal1, fxd->sz,
        cudaMemcpyHostToDevice);
    cudaMemcpy(fxd->cu_signal2, fxd->signal2, fxd->sz,
        cudaMemcpyHostToDevice);
    
    // transform signal1 *IN-PLACE*
    if(cufftExecC2C(
        fxd->cufft_plan, 
        fxd->cu_signal1, fxd->cu_signal1, 
        CUFFT_FORWARD) != CUFFT_SUCCESS )
    {
        fprintf(stderr, "Launching C2C FFT has failed.\n");
        return -1;
    }
    
    // transform signal2 *IN-PLACE*
    if(cufftExecC2C(
        fxd->cufft_plan, 
        fxd->cu_signal2, fxd->cu_signal2, 
        CUFFT_FORWARD) != CUFFT_SUCCESS )
    {
        fprintf(stderr, "Launching C2C FFT has failed.\n");
        return -1;
    }
    /*
    // calls above are asynchronous. explicit synchronization hurts
    // but it is needed this time.
    if(cudaThreadSynchronize() != cudaSuccess)
	{
		fprintf(stderr, "CUDA error: failed to synchronize!.\n"); 
        // if synchronization fails, it might still be possible to 
        // keep going, but I rather halt everything because it is an
        // indicator that something isn't working as intended.
        return -1;
	}
    */
    // multiply the coefficients together and normalize the result
    float T = 1.0/(2*fxd->nsamples - 1);
    // 32 -> block size of 32 (which is the warp size of most SMU 
    // in a variety of GPUs. 
    // 256 -> fine tune at will
    // result of this operation is held at fxd->cu_signal1
    complex_pointwise_cms<<<32, 256>>>(
        fxd->cu_signal1, fxd->cu_signal2, 2*fxd->nsamples, T);
    // transform back
    if(cufftExecC2C(
        fxd->cufft_plan, 
        fxd->cu_signal1, fxd->cu_signal1, 
        CUFFT_INVERSE) != CUFFT_SUCCESS)
    {
        fprintf( stderr, "Launching C2C inverse FFT has failed.\n" );
        return -1;
    }

    // copy fourier transform back to host
    cufftComplex* correlation = (cufftComplex*)malloc(
        sizeof(cufftComplex) * fxd->nsamples*2 );;
    cudaMemcpy(
        correlation, fxd->cu_signal1, 
        sizeof(cufftComplex)*fxd->nsamples*2, 
        cudaMemcpyDeviceToHost);

    // Copy correlation to host. First half in ascending order
    for( int i=0; i < fxd->nsamples ; i++ )
    {    
        fxd->out[i+fxd->nsamples] = correlation[fxd->nsamples+i].x/fxd->nsamples;
    }
    // Copy correlation to host. Second half in descending order
    for( int i=0, j=2*fxd->nsamples-1; i < fxd->nsamples ; i++,j-- )
    {    
        fxd->out[i] = correlation[j].x/fxd->nsamples;
    }    
    
    // free temporal buffer
    cudaFree(correlation);
    
    return 0;
}
