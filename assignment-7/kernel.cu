
#include "common.h"
#include "timer.h"


__global__ void scan_kernel(float* input, float* output, unsigned int N) {
    __shared__ float buffer[2*BLOCK_DIM];
    // For the reduction tree, we assign one thread for each pair of elements in the block.
    unsigned int i = blockIdx.x * 2*blockDim.x + threadIdx.x;
    if(i < N) {
        buffer[threadIdx.x] = input[i];
    }
    else{
        buffer[threadIdx.x] = 0.0f;
    }

    if( i + blockDim.x < N) {
        buffer[threadIdx.x + blockDim.x] = input[i + blockDim.x];
    }
    else{
        buffer[threadIdx.x + blockDim.x] = 0.0f;
    }
    
    __syncthreads();

    for(unsigned int stride = 1; stride < 2*blockDim.x; stride *= 2) {
        int offset = 2*stride*threadIdx.x;
        if(offset + stride < 2*blockDim.x && offset + 2*stride - 1 < 2*blockDim.x) {
            buffer[offset + 2*stride - 1] += buffer[offset + stride - 1];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0) {
        buffer[2*blockDim.x - 1] = 0.0f;
    }

    for(unsigned int stride = blockDim.x; stride > 1; stride /= 2) {
        int offset = 2*stride*threadIdx.x;
        if(offset + stride < 2*blockDim.x && offset + 2*stride - 1 < 2*blockDim.x) {
            float temp = buffer[offset + stride - 1];
            buffer[offset + stride - 1] = buffer[offset + 2*stride - 1];
            buffer[offset + 2*stride - 1] += temp;
        }
        __syncthreads();
    }

    if(i < N) {
        output[i] = buffer[threadIdx.x];
    }   
    if(i + blockDim.x < N) {
        output[i + blockDim.x] = buffer[threadIdx.x + blockDim.x];
    }
}

void scan_gpu_d(float* input_d, float* output_d, unsigned int N) {

    const unsigned int numThreadsPerBlock = BLOCK_DIM;
    const unsigned int numElementsPerBlock = 2*numThreadsPerBlock;
    const unsigned int numBlocks = (N + numElementsPerBlock - 1)/numElementsPerBlock;
    scan_kernel <<< numBlocks, numThreadsPerBlock >>> (input_d, output_d, N);

}

