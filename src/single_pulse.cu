#include <stdio.h>
#include <cuda.h>
#include <cufft.h>
#include <npp.h>
#include <math.h>
#include "cufft_error.h"

#define THREADS_PER_BLOCK 512

/**Taken from common.h from GPGPU Workshop code**/

#define CUDA_CALL(x,s) { cudaError_t rc = ( x ); if (rc != cudaSuccess) { \
        printf("%s (%s) at %s:%d\n", s, cudaGetErrorString(rc), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
}}

#define CUFFT_CALL(x,s) { cufftResult_t rc = ( x ); if (rc != CUFFT_SUCCESS ) { \
        printf("%s (%s) at %s:%d\n", s, cufftGetErrorString(rc), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
}}
   

void create_boxcar_kernel(float *kern, int filterWidth, int lenFFT)
{
    float *boxcars; 
    int jj=0;

    boxcars=(float *)malloc(sizeof(float)*lenFFT);
 
    printf("Filter width: %d\n", filterWidth);

    for(jj = 0; jj < lenFFT; jj++)
    {
        if(jj < filterWidth/2+1)
        {
            boxcars[jj]=1.0;
        }
        else if(jj>(lenFFT - filterWidth/2) && filterWidth%2 == 0 && filterWidth>2)
        {
            boxcars[jj]=1.0;
        }
        else if(jj>(lenFFT - filterWidth/2-1) && filterWidth%2 == 1)
        {
            boxcars[jj]=1.0;
        }
        else
            boxcars[jj]=0.0;
    }
    //Copy to GPU for now copy out of function
    memcpy(kern, boxcars, sizeof(float)*lenFFT);
    free(boxcars);
}

__global__ void complexMultiply(cufftComplex *fft_a, cufftComplex *fft_b, cufftComplex *fft_out, int fftlen)
{
    int ii = blockDim.x * blockIdx.x + threadIdx.x;

    if(ii < fftlen)
    {
        fft_out[ii].x = fft_a[ii].x * fft_b[ii].x - fft_a[ii].y * fft_b[ii].y;
        fft_out[ii].y = fft_a[ii].y * fft_b[ii].x + fft_a[ii].x * fft_b[ii].y;
    }
}

void convolve(cufftComplex *d_data, cufftComplex *d_box, int fftlen, cufftComplex *d_conv)
{
    cudaEvent_t start, stop;
    float RunTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cufftHandle plan;
    CUFFT_CALL(cufftPlan1d(&plan, fftlen, CUFFT_R2C, 1), 
        "Error generating plan for forward FFT");

    //In place transform
    CUFFT_CALL(cufftExecR2C(plan, (cufftReal *) d_box,  d_box), 
        "Error calculating forward FFT of filter kernel");
    CUFFT_CALL(cufftExecR2C(plan, (cufftReal *) d_data, d_data),
        "Error calculating forward FFT of TS data");

    CUFFT_CALL(cufftDestroy(plan), "Error destroying plan for forward FFT");

    //Multiply kernel and data
    cudaEventRecord(start, 0);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&RunTime, start, stop);
    printf("FFT length: %d \n", fftlen);
    printf("Configuring and launching the forward FFTs took %f ms \n", RunTime);

    //Set up and launch complex multiply thread
    cudaEventRecord(start, 0);
    int blocksPerGrid = (fftlen + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    complexMultiply <<< blocksPerGrid, THREADS_PER_BLOCK >>> (d_box, d_data, d_conv, fftlen);



    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&RunTime, start, stop);
    printf("The complex multiply took %f ms using %d threads on %d blocks\n", RunTime, THREADS_PER_BLOCK, blocksPerGrid);

    //Setup and launch inverse transform
    cudaEventRecord(start, 0);
    CUFFT_CALL(cufftPlan1d(&plan, fftlen, CUFFT_C2R, 1), 
        "Error generating plan for reverse FFT ");
    CUFFT_CALL(cufftExecC2R(plan, d_conv, (cufftReal *) d_conv), 
        "Error calculating reverse FFT");

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&RunTime, start, stop);
    printf("Configuring and launching the reverse FFT took %f ms \n", RunTime);
    CUFFT_CALL(cufftDestroy(plan), "Error destroying reverse FFT plan");
}

void calc_stats(float *indata, int size, float *tsmean, float *tsstd)
{
    NppStatus npp_status;

    //Determine buffer size
    int BuffSize;
    nppsMeanStdDevGetBufferSize_32f(size, &BuffSize); 
    Npp8u *pDevBuff;

    float *d_mean, *d_std;
    float mtmp[1]={-2.0};
    float stmp[1]={-2.0};

    printf("M & S inside: %f %f\n", mtmp, stmp);

    CUDA_CALL(cudaMalloc((void **) &d_mean, sizeof(float)), "Allocate device float");
    CUDA_CALL(cudaMalloc((void **) &d_std, sizeof(float)), "Allocate device float");
    CUDA_CALL(cudaMemcpy(d_mean, mtmp, sizeof(float), cudaMemcpyHostToDevice), "CPY");
    CUDA_CALL(cudaMemcpy(d_std, stmp, sizeof(float), cudaMemcpyHostToDevice), "CPY");

    printf("Buff size: %d\n", BuffSize);
    //Allocate scratch buffer
    CUDA_CALL(cudaMalloc((void **) &pDevBuff, BuffSize), 
        "Failure allocating scratch buffer for stats calc.");

    //Calc stats
    npp_status=nppsMeanStdDev_32f( (Npp32f *) indata, size, d_mean, d_std, pDevBuff);
    printf("Status: %d\n", npp_status);

    CUDA_CALL(cudaMemcpy(tsmean, d_mean, sizeof(float), cudaMemcpyDeviceToHost), 
        "Copy mean back");
    CUDA_CALL(cudaMemcpy(tsstd, d_std, sizeof(float), cudaMemcpyDeviceToHost), 
        "Copy stddev back");

    printf("M & S inside: %f %f\n", tsmean, tsstd);

    //Free memory
    CUDA_CALL(cudaFree(d_mean), "Freeing mean array");
    CUDA_CALL(cudaFree(d_std), "Freeing stddev array");
    CUDA_CALL(cudaFree(pDevBuff), "Freeing buffer memory");    
}

extern "C" {

 void ts_convolve(float *indata, int width, int fftlen, float *outdata)
 {
    size_t size = fftlen*sizeof(float);

    cudaEvent_t start, stop; 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *boxout = (float *)malloc(size);

    //Print some device properties
    cudaDeviceProp dev_prop;
    CUDA_CALL(cudaGetDeviceProperties(&dev_prop, 0), "Error getting device properties");

    printf("******** DEVICE PROPERTIES ********\n");
    printf("Device name: %s\n", dev_prop.name);
    printf("Maximum Threads per Block: %d\n", dev_prop.maxThreadsPerBlock);
    printf("Maximum Dimensions of a Block: %d %d %d\n", dev_prop.maxThreadsDim[0],
       dev_prop.maxThreadsDim[1], dev_prop.maxThreadsDim[2]);
    printf("Maximum Dimensions of a Grid: %d %d %d\n", 
        dev_prop.maxGridSize[0], dev_prop.maxGridSize[1], dev_prop.maxGridSize[2]);
    printf("Warp size in threads: %d\n", dev_prop.warpSize);
    printf("Shared Memory per Block in Bytes: %d\n", dev_prop.sharedMemPerBlock);
    printf("***********************************\n\n");

    //Create boxcar kernal array
    create_boxcar_kernel(boxout, width, fftlen);
 
    //Copy kernel to GPU (eventually loop over several matched filter kernels
    cufftComplex *d_box = NULL;
    cufftComplex *d_data = NULL;
    cufftComplex *d_conv = NULL;

    int compSize = sizeof(cufftComplex)*fftlen;

    CUDA_CALL(cudaMalloc((void **)&d_box, compSize), "Error allocating array d_box");
    CUDA_CALL(cudaMalloc((void **)&d_data, compSize), "Error allocating array d_data"); 
    CUDA_CALL(cudaMalloc((void **)&d_conv, compSize), "Error allocating array d_conv");

    CUDA_CALL(cudaMemcpy(d_box, boxout, size, cudaMemcpyHostToDevice), 
        "Error copying filter kernel data to GPU");
    CUDA_CALL(cudaMemcpy(d_data, indata, size, cudaMemcpyHostToDevice), 
        "Error TS data to GPU");

    //Calculate convolution
    convolve(d_data,  d_box, fftlen, d_conv);

    //Copy back 
    CUDA_CALL(cudaMemcpy(outdata, d_conv, size, cudaMemcpyDeviceToHost), 
        "Error copying convolved TS from GPU");

    free(boxout);

    CUDA_CALL(cudaFree(d_box), "Error freeing d_box");
    CUDA_CALL(cudaFree(d_data), "Error freeing d_data");
//    CUDA_CALL(cudaFree(d_conv), "Error freeing d_conv");

//    cudaDeviceReset();

    //Calc stats
    float tsmean[1]={0.0};
    float tsstd[1]={0.0};

    calc_stats((float *) d_conv, fftlen, tsmean, tsstd);
    printf("Mean and Std Dev: %f %f\n", tsmean[0], tsstd[0]);

    CUDA_CALL(cudaFree(d_conv), "Error freeing d_conv");
 }
}
