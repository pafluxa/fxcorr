#include <math.h>
#include <time.h>
#include <stdio.h>

#include "fxcorr.cuh"

int main(void)
{
    fxcorrData* fxd = (fxcorrData*)malloc(sizeof(fxcorrData));
    // 67108864 samples
    long nsamples = 1 << 26;
    printf("#N = %ld\n", nsamples);
    float* s1 = (float*)malloc(4*sizeof(float)*2*nsamples);
    float* s2 = (float*)malloc(4*sizeof(float)*2*nsamples);
    for(long i=0; i<nsamples; i++)
    {
        s1[2*i] = exp(-0.5*(i - 200)*(i - 200)/(10*10));
        s1[2*i + 1] = 0.0;
        
        s2[2*i] = exp(-0.5*(i - 1000)*(i - 1000)/(10*10));
        s2[2*i + 1] = 0.0;
    }

    clock_t t;
    t = clock();

    fxcorr_attach_signals(s1, s2, nsamples, fxd);
    fxcorr_allocate_and_zeropad(fxd);
    fxcorr_create_cufft_handle(fxd);
    fxcorr_compute(fxd);\
    // uncomment to print result to stdout
    //for(long i=0; i<nsamples; i++){printf("%06ld%10.2E\n", i, fxd->out[i]);}
    fxcorr_deallocate(fxd);

    t = clock() - t;
    // calculate the elapsed time
    double time_taken = ((double)t)/CLOCKS_PER_SEC; 
    printf("correlation took %f seconds to execute\n", time_taken);
    
    
    return 0;
}
