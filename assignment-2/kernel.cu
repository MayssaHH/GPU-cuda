
#include "common.h"
#include "timer.h"

__global__ void mm_kernel(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;


    if (row < M && col < N) {

        float sum = 0.0f;
        for (unsigned int i = 0; i < K; ++i) {
            sum += A[row*K + i]*B[i*N + col];
        }
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

    float* A_gpu, *B_gpu, *C_gpu;
    cudaMalloc((void**)&A_gpu, M*K*sizeof(float));
    cudaMalloc((void**)&B_gpu, K*N*sizeof(float));
    cudaMalloc((void**)&C_gpu, M*N*sizeof(float));


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Allocation time: %.3f ms\n", elapsedTime);

    // Copy data to GPU
    cudaEventRecord(start);

    cudaMemcpy(A_gpu, A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, K*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_gpu, C, M*N*sizeof(float), cudaMemcpyHostToDevice);


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Copy to GPU time: %.3f ms\n", elapsedTime);

    // Call kernel
    cudaEventRecord(start);

    dim3 numThreadsperBlock(32, 32);
    dim3 numBlocks((N+numThreadsperBlock.x-1)/numThreadsperBlock.x, (M+numThreadsperBlock.y-1)/numThreadsperBlock.y);
    
    mm_kernel<<< numBlocks, numThreadsperBlock >>> (A_gpu, B_gpu, C_gpu, M, N, K);


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\033[1;32mKernel time: %.3f ms\033[0m\n", elapsedTime);

    // Copy data from GPU
    cudaEventRecord(start);

    cudaMemcpy(C, C_gpu, M*N*sizeof(float), cudaMemcpyDeviceToHost);


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Copy from GPU time: %.3f ms\n", elapsedTime);

    // Free GPU memory
    cudaEventRecord(start);

    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Deallocation time: %.3f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}

