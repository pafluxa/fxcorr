#include <math.h>
#include <time.h>
#include <stdio.h>
#include <fftw3.h>

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

    fxcorr_attach_signals(s1, s2, nsamples, fxd);
    fxcorr_allocate_and_zeropad(fxd);
    fxcorr_create_cufft_handle(fxd);
    clock_t t;
    t = clock();
    fxcorr_compute(fxd);
    // uncomment to print result to stdout
    //for(long i=0; i<nsamples; i++){printf("%06ld%10.2E\n", i, fxd->out[i]);}
    fxcorr_deallocate(fxd);
    
    t = clock() - t;
    // calculate the elapsed time
    double time_taken = ((double)t)/CLOCKS_PER_SEC; 
    printf("correlation took %f seconds on GPU\n", time_taken);
    
    fftw_complex* ss1 = fftw_alloc_complex(nsamples);
    fftw_complex* ss2 = fftw_alloc_complex(nsamples);
    fftw_complex* sr1 = fftw_alloc_complex(nsamples);
    fftw_complex* sr2 = fftw_alloc_complex(nsamples);

    fftw_plan_with_nthreads(16);
    fftw_plan plan1 = fftw_plan_dft_1d(
        nsamples,
        ss1,
        sr1,
        FFTW_FORWARD,
        FFTW_ESTIMATE);

    fftw_plan plan2 = fftw_plan_dft_1d(
        nsamples,
        ss2,
        sr2,
        FFTW_FORWARD,
        FFTW_ESTIMATE);

    fftw_plan plan3 = fftw_plan_dft_1d(
        nsamples,
        sr2,
        ss2,
        FFTW_BACKWARD,
        FFTW_ESTIMATE);

    for(long i=0; i<nsamples; i++)
    {
        ss1[i][0] = s1[2*i];
        ss1[i][1] = s1[2*i+1];
        
        ss2[i][0] = s2[2*i];
        ss2[i][1] = s2[2*i + 1];
    }
    t = clock();
    fftw_execute(plan1);
    fftw_execute(plan2);
    fftw_execute(plan3);
    t = clock() - t;
    // calculate the elapsed time
    time_taken = ((double)t)/CLOCKS_PER_SEC; 
    printf("correlation took %f seconds on CPU\n", time_taken);

    fftw_destroy_plan(plan1);
    fftw_destroy_plan(plan2);
    fftw_destroy_plan(plan3);
    
    
    return 0;
}
