
#include "common.h"
#include "timer.h"

#define TILE_DIM 32

// I assume that A is MxK and B is KxN and C is MxN
__global__ void mm_tiled_kernel(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

    // First, I have to declare the shared memory for the tiles of A and B
    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];

    // Then, I have to declare the variables for the row and column of the thread
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    float sum = 0.0f;

    // Now, i need to iterate over the tiles, which are (K+TILE_DIM-1)/TILE_DIM
    for (unsigned int tile = 0; tile < (K+TILE_DIM-1)/TILE_DIM; ++tile) {
        // Load tiles to shared memory 
        if(row < M && tile*TILE_DIM + threadIdx.x < K) {
            A_s[threadIdx.y][threadIdx.x] = A[row*K + tile*TILE_DIM + threadIdx.x];
        } else {
            A_s[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if(col < N && tile*TILE_DIM + threadIdx.y < K) {
            B_s[threadIdx.y][threadIdx.x] = B[(tile*TILE_DIM + threadIdx.y)*N + col];
        } else {
            B_s[threadIdx.y][threadIdx.x] = 0.0f;
        }
        // Threads wait for each other to finish loading before computing
        __syncthreads();

        // Compute using the tiles 
        for (unsigned int i = 0; i < TILE_DIM; ++i) {
            sum += A_s[threadIdx.y][i]*B_s[i][threadIdx.x];
        }
        // Threads wait for each other to finish computing before loading the next tile
        __syncthreads();
    }
    if(row < M && col < N) {
        C[row*N + col] = sum;
    }
}

void mm_gpu(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate GPU memory
    cudaEventRecord(start);

    float* A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, M*K*sizeof(float));
    cudaMalloc((void**)&B_d, K*N*sizeof(float));
    cudaMalloc((void**)&C_d, M*N*sizeof(float));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Allocation time: %.3f ms\n", elapsedTime);

    // Copy data to GPU
    cudaEventRecord(start);

    cudaMemcpy(A_d, A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, K*N*sizeof(float), cudaMemcpyHostToDevice);


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Copy to GPU time: %.3f ms\n", elapsedTime);

    // Call kernel
    cudaEventRecord(start);

    dim3 numThreadsperBlock(TILE_DIM, TILE_DIM);
    dim3 numBlocks((N+numThreadsperBlock.x-1)/numThreadsperBlock.x, (M+numThreadsperBlock.y-1)/numThreadsperBlock.y);

    mm_tiled_kernel <<< numBlocks, numThreadsperBlock >>> (A_d, B_d, C_d, M, N, K);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\033[1;32mKernel time: %.3f ms\033[0m\n", elapsedTime);

    // Copy data from GPU
    cudaEventRecord(start);
    cudaMemcpy(C, C_d, M*N*sizeof(float), cudaMemcpyDeviceToHost);


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Copy from GPU time: %.3f ms\n", elapsedTime);

    // Free GPU memory
    cudaEventRecord(start);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Deallocation time: %.3f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}

