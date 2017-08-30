/* Example showing the use of CUFFT for fast 1D-convolution using FFT. */
////////////////////////////////////////////////////////////////////////////////

#if __cplusplus
extern "C" {
#endif

void cudacorrelate_fft
( 
    float re_signal1[], float im_signal1[], 
    float re_signal2[], float im_signal2[], int nsamples,
    // output
    float cs1s2[]
);

#if __cplusplus
}
#endif
