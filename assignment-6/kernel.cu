#include "common.h"

#include "timer.h"

#include <cooperative_groups.h>
using namespace cooperative_groups;

#define BLOCK_DIM 1024
#define WARP_SIZE 32

__device__ inline int laneIdx() { return threadIdx.x % WARP_SIZE; }

__device__ __inline__ float warp_reduce(float val) {
    thread_block_tile<WARP_SIZE> tile = tiled_partition<WARP_SIZE>(this_thread_block());
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += tile.shfl_down(val, offset);
    }
    return val;
}

__global__ void reduce_kernel(float* input, float* sum, unsigned int N) {
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float partialSum = 0.0f;
    if (idx < N) {
        partialSum += input[idx];
    }
    if (idx + blockDim.x < N) {
        partialSum += input[idx + blockDim.x];
    }

    partialSum = warp_reduce(partialSum);

    __shared__ float partialSums_s[BLOCK_DIM / WARP_SIZE];
    if (laneIdx() == 0) {
        partialSums_s[threadIdx.x / WARP_SIZE] = partialSum;
    }
    __syncthreads();

    if (threadIdx.x < BLOCK_DIM / WARP_SIZE) {
        float val = partialSums_s[threadIdx.x];
        val = warp_reduce(val);
        if (threadIdx.x == 0) {
            atomicAdd(sum, val);
        }
    }
}

float reduce_gpu(float* input, unsigned int N) {

    Timer timer;

    // Allocate memory
    startTime(&timer);
    float *input_d;
    cudaMalloc((void**) &input_d, N*sizeof(float));
    float *sum_d;
    cudaMalloc((void**) &sum_d, sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);
    cudaMemcpy(input_d, input, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(sum_d, 0, sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Call kernel
    startTime(&timer);
    const unsigned int numThreadsPerBlock = BLOCK_DIM;
    const unsigned int numElementsPerBlock = 2*numThreadsPerBlock;
    const unsigned int numBlocks = (N + numElementsPerBlock - 1)/numElementsPerBlock;
    reduce_kernel <<< numBlocks, numThreadsPerBlock >>> (input_d, sum_d, N);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    // Copy data from GPU
    startTime(&timer);
    float sum;
    cudaMemcpy(&sum, sum_d, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Free memory
    startTime(&timer);
    cudaFree(input_d);
    cudaFree(sum_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

    return sum;

}

