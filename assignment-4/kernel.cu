
#include "common.h"
#include "timer.h"

#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))

__constant__ float filter_c[FILTER_DIM][FILTER_DIM];

__global__ void convolution_tiled_kernel(float* input, float* output, unsigned int width, unsigned int height) {

    // First, I have to declare the shared memory for the tile
    __shared__ float tile[IN_TILE_DIM][IN_TILE_DIM];

    // Since BlockDim = IN_TILE_DIM, we will locate output tile then go back to input tile
    // outRow_tile = blockIdx.y*OUT_TILE_DIM + threadIdx.y
    // outCol_tile = blockIdx.x*OUT_TILE_DIM + threadIdx.x
    int inRow_tile = blockIdx.y*OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
    int inCol_tile = blockIdx.x*OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;

    // Now load the input tile into shared memory with boundary conditions
    if (inRow_tile >= 0 && inRow_tile < height && inCol_tile >= 0 && inCol_tile < width) {
        tile[threadIdx.y][threadIdx.x] = input[inRow_tile*width + inCol_tile];
    } else {
        tile[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();
    
    // Each thread in [0, OUT_TILE_DIM) x [0, OUT_TILE_DIM) computes one output using the loaded tile 
    if (threadIdx.x < OUT_TILE_DIM && threadIdx.y < OUT_TILE_DIM) {
        int outRow = blockIdx.y*OUT_TILE_DIM + threadIdx.y;
        int outCol = blockIdx.x*OUT_TILE_DIM + threadIdx.x;
        float sum = 0.0f;
        if (outRow < height && outCol < width) {
            for(int filterRow = 0; filterRow < FILTER_DIM; ++filterRow) {
                for(int filterCol = 0; filterCol < FILTER_DIM; ++filterCol) {
                    sum += filter_c[filterRow][filterCol]*tile[threadIdx.y + filterRow][threadIdx.x + filterCol];
                }
            }
            output[outRow*width + outCol] = sum;
        }
    }
}

void copyFilterToGPU(float filter[][FILTER_DIM]) {
    // Copy filter to constant memory
    cudaMemcpyToSymbol(filter_c, filter, FILTER_DIM*FILTER_DIM*sizeof(float));
}

void convolution_tiled_gpu(float* input_d, float* output_d, unsigned int width, unsigned int height) {

    // Call kernel
    dim3 numThreadsPerBlock(IN_TILE_DIM, IN_TILE_DIM);
    dim3 numBlocks((width + OUT_TILE_DIM - 1)/OUT_TILE_DIM, (height + OUT_TILE_DIM - 1)/OUT_TILE_DIM);

    convolution_tiled_kernel <<< numBlocks, numThreadsPerBlock >>> (input_d, output_d, width, height);
}

