
#include "common.h"
#include "timer.h"

#include <cuda/atomic>

#define COARSE_FACTOR 64

__global__ void histogram_private_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
    
    __shared__ unsigned int bins_shared[NUM_BINS];

    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int binIndex = threadIdx.x;

    if(binIndex < NUM_BINS) {
        bins_shared[binIndex] = 0;
    }

    __syncthreads();

    if(index < width*height) {
        unsigned char b = image[index];
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_shared_ref(bins_shared[b]);
        bins_shared_ref.fetch_add(1, cuda::memory_order_relaxed);
    }

    __syncthreads();  

    if(binIndex < NUM_BINS) {
        if(bins_shared[binIndex] > 0) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[binIndex]);
            bins_ref.fetch_add(bins_shared[binIndex], cuda::memory_order_relaxed);
        }
    }
}

void histogram_gpu_private(unsigned char* image_d, unsigned int* bins_d, unsigned int width, unsigned int height) {

    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (width*height + numThreadsPerBlock - 1)/numThreadsPerBlock;
    histogram_private_kernel <<< numBlocks, numThreadsPerBlock >>> (image_d, bins_d, width, height);

}

__global__ void histogram_private_coarse_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {

    __shared__ unsigned int bins_shared[NUM_BINS];

    int index = blockIdx.x*blockDim.x*COARSE_FACTOR + threadIdx.x;
    int binIndex = threadIdx.x;

    if(binIndex < NUM_BINS) {
        bins_shared[binIndex] = 0;
    }

    __syncthreads();

    if(index < width*height) {
        for(unsigned int i = 0; i < COARSE_FACTOR; ++i) {
            unsigned int in = index + i*blockDim.x;
            if(in < width*height) {
                unsigned char b = image[in];
                cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_shared_ref(bins_shared[b]);
                bins_shared_ref.fetch_add(1, cuda::memory_order_relaxed);
            }
        }
    }

    __syncthreads();  

    if(binIndex < NUM_BINS) {
        if(bins_shared[binIndex] > 0) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[binIndex]);
            bins_ref.fetch_add(bins_shared[binIndex], cuda::memory_order_relaxed);
        }
    }
}

void histogram_gpu_private_coarse(unsigned char* image_d, unsigned int* bins_d, unsigned int width, unsigned int height) {

    unsigned int numThreadsPerBlock = 1024;
    unsigned int numElementsPerBlock = COARSE_FACTOR*numThreadsPerBlock;
    unsigned int numBlocks = (width*height + numElementsPerBlock - 1)/numElementsPerBlock;
    histogram_private_coarse_kernel <<< numBlocks, numThreadsPerBlock >>> (image_d, bins_d, width, height);

}

