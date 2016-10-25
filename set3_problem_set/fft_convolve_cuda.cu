/* CUDA blur
 * Kevin Yuh, 2014 */

#include <cstdio>

#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_convolve_cuda.cuh"


/* 
Atomic-max function. You may find it useful for normalization.

We haven't really talked about this yet, but __device__ functions not
only are run on the GPU, but are called from within a kernel.

Source: 
http://stackoverflow.com/questions/17399119/
cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
*/
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}



__global__
void
cudaProdScaleKernel(const cufftComplex *raw_data, const cufftComplex *impulse_v, 
    cufftComplex *out_data,
    int padded_length) {

   unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < padded_length) {
        cufftComplex a = raw_data[index];
        cufftComplex b = impulse_v[index];
        cufftComplex c;
        c.x = ((a.x * b.x) - (a.y * b.y)) / padded_length;
        c.y = ((a.x * b.y) + (a.y * b.x)) / padded_length;
        
        out_data[index] = c;
        
        index += blockDim.x * gridDim.x;
	}
}



__global__
void
cudaMaximumKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {
    float max_abs_vals[1024];
    
    int tid =  blockIdx.x * blockDim.x + threadIdx.x;
    int index = tid +  blockDim.x;
    max_abs_vals[tid] = out_data[tid];
    while( index < padded_length){
   	atomicMax(&max_abs_vals[tid], out_data[index];
	index += blockDim.x; 
    }
    __syncthreads();
    int s = 2;
    while( blockDim.x / s > 32){
  	
	if( tid <  blockDim.x /s)
		atomixMax(&max_abs_vals[tid], max_abs+vals[2*tid]);
	s *= 2;
	__syncthreads(); 

    }

	:
}

__global__
void
cudaDivideKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* TODO 2: Implement the division kernel. Divide all
    data by the value pointed to by max_abs_val. 

    This kernel should be quite short.
    */

}


void cudaCallProdScaleKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        const cufftComplex *impulse_v,
        cufftComplex *out_data,
        const unsigned int padded_length) {
	cudaProdScaleKernel<<<blocks, threadsPerBlock >>>(raw_data, impulse_v, out_data, padded_length);
}

void cudaCallMaximumKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        

    /* TODO 2: Call the max-finding kernel. */

}


void cudaCallDivideKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        
    /* TODO 2: Call the division kernel. */
}
