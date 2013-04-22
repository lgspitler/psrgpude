#define MAX_BLOCKS 65535
#define MAX_THREADS 512
#include <iostream>
using namespace std;
/*
__global__ void harmonic_sum_kernel(float *d_idata,int gulp_index, int size, int stretch_factor)
{
    //float* d_idata_float = (float*)d_idata;
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index<size/stretch_factor)
    {
        for(int i = 0;i<stretch_factor;i++)
        {
            d_idata[gulp_index+stretch_factor*Index+i] =
                                      d_idata[gulp_index+stretch_factor*Index+i]
                                     +d_idata[gulp_index+Index];
        }
    }
return;
}


__global__ void harmonic_sum_kernel_16(float *d_idata, float *d_odata,int gulp_index, int size)
{
    //float* d_idata_float = (float*)d_idata;
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index<size)
    {
        d_odata[gulp_index+Index] = 


                        d_idata[1*(gulp_index+Index)/2]
                        +d_idata[gulp_index+Index]

                        +d_idata[1*(gulp_index+Index)/4]
                        +d_idata[3*(gulp_index+Index)/4]

                        +d_idata[1*(gulp_index+Index)/8]
                        +d_idata[3*(gulp_index+Index)/8]
                        +d_idata[5*(gulp_index+Index)/8]
                        +d_idata[7*(gulp_index+Index)/8]

                        +d_idata[(gulp_index+Index)/16]
                        +d_idata[3*(gulp_index+Index)/16]
                        +d_idata[5*(gulp_index+Index)/16]
                        +d_idata[7*(gulp_index+Index)/16]
                        +d_idata[9*(gulp_index+Index)/16]
                        +d_idata[11*(gulp_index+Index)/16]
                        +d_idata[13*(gulp_index+Index)/16]
                        +d_idata[15*(gulp_index+Index)/16];
    }
return;
}

__global__ void harmonic_sum_kernel_8(float *d_idata, float *d_odata,int gulp_index, int size)
{
    //float* d_idata_float = (float*)d_idata;
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index<size)
    {
        d_odata[gulp_index+Index] = d_idata[(gulp_index+Index)/8]
                        +d_idata[2*(gulp_index+Index)/8]
                        +d_idata[3*(gulp_index+Index)/8]
                        +d_idata[4*(gulp_index+Index)/8]
                        +d_idata[5*(gulp_index+Index)/8]
                        +d_idata[6*(gulp_index+Index)/8]
                        +d_idata[7*(gulp_index+Index)/8]
                        +d_idata[gulp_index+Index];
    }
return;
}

__global__ void harmonic_sum_kernel_4(float *d_idata, float *d_odata,int gulp_index, int size)
{
    //float* d_idata_float = (float*)d_idata;
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index<size)
    {
        d_odata[gulp_index+Index] = d_idata[(gulp_index+Index)/4]
                        +d_idata[2*(gulp_index+Index)/4]
                        +d_idata[3*(gulp_index+Index)/4]
                        +d_idata[gulp_index+Index];
    }
return;
}

__global__ void harmonic_sum_kernel_2(float *d_idata, float *d_odata,int gulp_index, int size)
{
    //float* d_idata_float = (float*)d_idata;
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index<size)
    {
        d_odata[gulp_index+Index] = d_idata[(gulp_index+Index)/2]
                        +d_idata[gulp_index+Index];
    }
return;
}


void call_harmonic_sum_kernel_generic(float *d_idata, float *d_odata, int gulp_index, int size, int harmonic)
{
   harmonic_sum_kernel_generic(d_idata, d_odata, gulp_index, size, harmonic);
}
*/

__global__ void harmonic_sum_kernel_generic(float *d_idata, float *d_odata,int gulp_index, int size,int harmonic)
{
    //float* d_idata_float = (float*)d_idata;
    int Index = blockIdx.x * blockDim.x + threadIdx.x;
    if(Index<size)
    {
        d_odata[gulp_index+Index] = d_idata[gulp_index+Index];
        for(int i = 1; i < harmonic; i++)
        {
            d_odata[gulp_index+Index] += d_idata[(i*(gulp_index+Index))/harmonic];
        }
// NOTE ERROR HERE
        d_odata[gulp_index+Index] = d_odata[gulp_index+Index]/sqrt((float)harmonic);  // can use *rsqrt to optimise further
    }
return;
}

void GPU_harmonic_sum(float* d_input_array, float* d_output_array, int original_size, int harmonic)
{
    int gulps;
    int gulp_counter;
    int gulp_index = 0;
    int gulp_size;

    gulps = original_size/(MAX_BLOCKS*MAX_THREADS)+1;

    for(gulp_counter = 0; gulp_counter<gulps; gulp_counter++)
    {
        if(gulp_counter<gulps-1)
        {
            gulp_size = MAX_BLOCKS*MAX_THREADS;
        }
        else
        {
            gulp_size = original_size - gulp_counter*MAX_BLOCKS*MAX_THREADS;
            
        }
        harmonic_sum_kernel_generic<<<MAX_BLOCKS,MAX_THREADS>>>(d_input_array,d_output_array,gulp_index,gulp_size,harmonic);
        gulp_index = gulp_index + MAX_BLOCKS*MAX_THREADS;
    }

    return;
}
