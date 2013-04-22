#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cufft.h>
#include <math.h>
#define MAX_BLOCKS 65535
#define MAX_THREADS 512
#define THREADS_PER_BLOCK 1024

using namespace std;

/*--------------------------------------------------------------*/
inline __device__ int getAcceleratedIndex(float, int, float, int);
__global__ void resampleOnDevice(float*, float*, float, int, float);
__global__ void harmonicSumOnDevice(float*, float*, int, int, int);
__global__ void formSpecOnDevice(float*, float*);
void GPU_harmonic_sum(float*, float*, int, int);
extern "C"{
  int checkError(void);
  int seekGPU(float*, size_t, float*, size_t, float);
}
/*-------------------------------------------------------------*/


inline __device__ int getAcceleratedIndex(float accel_fact, int size_by_2, int id){
  return (int)(id + accel_fact*( ((id-size_by_2)*(id-size_by_2)) - (size_by_2*size_by_2)));
}

__global__ void resampleOnDevice(float* input_d,
				 float* output_d,
				 float accel_fact,
				 int size,
				 float size_by_2)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  int index0 = getAcceleratedIndex(accel_fact,size_by_2,id);
  int index1 = getAcceleratedIndex(accel_fact,size_by_2,id+1);
  output_d[index0] = input_d[id];
  if (index1-index0 > 1){
    if (index0+1 < size)
    output_d[index0+1] = input_d[id];
  }
}

__global__ void harmonicSumOnDevice(float *d_idata,
				    float *d_odata,
				    int gulp_index,
				    int size,
				    int harmonic)
{
  int ii;
  int Index = blockIdx.x * blockDim.x + threadIdx.x;
  if(Index<size){
    d_odata[gulp_index+Index] = d_idata[gulp_index+Index];
    for(ii=1; ii<harmonic; ii++){
      d_odata[gulp_index+Index] += d_idata[(ii*(gulp_index+Index))/harmonic];
    }
    d_odata[gulp_index+Index] = d_odata[gulp_index+Index]/sqrt((float)harmonic);  // can use *rsqrt to optimise further                               
  }
}

__global__ void formSpecOnDevice(float* f_spectrum_d,
				 float* p_spectrum_d)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  float i,r,a,b,rl,il;
  r = f_spectrum_d[2*id];
  i = f_spectrum_d[2*id+1];
  a = r*r + i*i;
  if (id == 0){
    rl = 0;
    il = 0;
  } else {
    rl = f_spectrum_d[2*(id-1)];
    il = f_spectrum_d[2*(id-1)+1];
  }
  a = r*r + i*i ;    
  b = ((r-rl)*(r-rl) + (i-il)*(i-il))/2 ;
  p_spectrum_d[id] = rsqrtf(fmax(a,b));
}

void GPU_harmonic_sum(float* d_input_array,
		      float* d_output_array,
		      int original_size,
		      int harmonic)
{
  int gulps;
  int gulp_counter;
  int gulp_index = 0;
  int gulp_size;
  gulps = original_size/(MAX_BLOCKS*MAX_THREADS)+1;
  for(gulp_counter = 0; gulp_counter<gulps; gulp_counter++){
    if(gulp_counter<gulps-1){
      gulp_size = MAX_BLOCKS*MAX_THREADS;
    } else {
      gulp_size = original_size - gulp_counter*MAX_BLOCKS*MAX_THREADS;
    }
    harmonicSumOnDevice<<<MAX_BLOCKS,MAX_THREADS>>>(d_input_array,d_output_array,gulp_index,gulp_size,harmonic);
    gulp_index = gulp_index + MAX_BLOCKS*MAX_THREADS;
  }
}

extern "C" {

  int checkError(void){
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err){
      fprintf(stderr,"CUDA error: %s\n",cudaGetErrorString(err));
      return(-1);}
    err = cudaDeviceSynchronize();
    if (cudaSuccess != err){
      fprintf(stderr,"CUDA error: %s\n",cudaGetErrorString(err));
      return(-1);}
    return 0;
  }

  int seekGPU(float* timeseries_h,
	      size_t size,
	      float* accels,
	      size_t naccels,
	      float  tsamp)
	      
  {
    int ii;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;
    int nharms = 5;

    //Print some device properties
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, 0);

    printf("Device name: %s\n", dev_prop.name);
    printf("Maximum Threads per Block: %d\n", dev_prop.maxThreadsPerBlock);
    printf("Maximum Dimensions of a Block: %d %d %d\n", dev_prop.maxThreadsDim[0],
       dev_prop.maxThreadsDim[1], dev_prop.maxThreadsDim[2]);
    printf("Maximum Dimensions of a Grid: %d %d %d\n", 
        dev_prop.maxGridSize[0], dev_prop.maxGridSize[1], dev_prop.maxGridSize[2]);
    printf("Warp size in threads: %d\n", dev_prop.warpSize);
    printf("Shared Memory per Block in Bytes: %d\n", dev_prop.sharedMemPerBlock);
    printf("\n");

    cudaEventRecord(start, 0);
    cufftHandle  plan;
    cufftPlan1d(&plan, size, CUFFT_R2C, 1);
    checkError();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time to create plan:  %f ms\n", elapsedTime);
    
    cudaEventRecord(start, 0);
    cufftReal    *timeseries_d;
    cufftReal    *resampled_d;
    cufftComplex *f_spectrum_d;
    cufftReal    *p_spectrum_d;
    cufftReal    *p_harmonics_d[nharms];
    
    size_t real_size    = sizeof(cufftReal)*size;
    size_t complex_size = sizeof(cufftComplex)*(size/2+1);
    
    cudaMalloc((void**)&timeseries_d, real_size);
    cudaMalloc((void**)&resampled_d,  real_size);
    cudaMalloc((void**)&f_spectrum_d, complex_size);
    cudaMalloc((void**)&p_spectrum_d, complex_size);
    
    for (ii=0;ii<nharms;ii++){
      cudaMalloc((void**)&p_harmonics_d[ii], real_size);
    }
    
    checkError();
    
    cudaMemcpy(timeseries_d, timeseries_h, real_size, cudaMemcpyHostToDevice);
    checkError();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time to allocate memory and copy:  %f ms\n", elapsedTime);

    cudaEventRecord(start, 0);
    dim3 grid(size/THREADS_PER_BLOCK,1,1);
    
    int size_by_2 = size/2;
    float accel_fact;

    printf("Size of data: %d\n", size);
    printf("Launching %d threads on (%d,%d,%d) blocks\n", THREADS_PER_BLOCK, grid.x, grid.y, grid.z);
    for (ii=0;ii<naccels;ii++){
      accel_fact = ((accels[ii]*tsamp) / (2 * 299792458.0));
      resampleOnDevice<<<grid, THREADS_PER_BLOCK>>>((float*) timeseries_d, (float*) resampled_d, accel_fact, size, size_by_2);
      cufftExecR2C(plan, (cufftReal *)resampled_d, (cufftComplex *)f_spectrum_d);
      formSpecOnDevice<<<grid,THREADS_PER_BLOCK>>>((float*) f_spectrum_d, (float*) p_spectrum_d);
      //for (ii=0;ii<nharms;ii++){
      //GPU_harmonic_sum(p_spectrum_d, p_harmonics_d[ii], complex_size, ii); NEEDS NEW HARMSUM ROUTINE
      //}
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time to execute:  %f ms\n", elapsedTime);
    checkError();
    //cudaMemcpy(outbuffer, odata, complex_size, cudaMemcpyDeviceToHost);

    //This is useful if you want to debug results
    cudaMemcpy(timeseries_h, p_spectrum_d, real_size, cudaMemcpyDeviceToHost);

    checkError();

    cufftDestroy(plan);
    cudaFree(timeseries_d);
    cudaFree(resampled_d);
    cudaFree(f_spectrum_d);
    cudaFree(p_spectrum_d);
    for (ii=0;ii<nharms;ii++){
      cudaFree(p_harmonics_d[ii]);
    }
    cudaDeviceReset();
    return checkError();
  }
}

