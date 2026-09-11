#include "mse.h"



__global__ void mse_kernel(float* A, float* B, float* C, int vectorLength) {
    __shared__ float partialSums[256];

    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;
    float difference = 0.0f;

    if(workIndex < vectorLength)
    {
        difference = A[workIndex] - B[workIndex];
    }

    partialSums[threadIdx.x] = difference * difference;
    __syncthreads();

    // cool tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partialSums[threadIdx.x] += partialSums[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(C, partialSums[0] / vectorLength);
    }
}

void launch_mse_kernel(float* A, float* B, float* C, int size) {
    // assumes A and B are already on device
    CUDA_CHECK(cudaMemset(C, 0, sizeof(float)));

    int threads = 256;
    int blocks = cuda::ceil_div(size, threads);

    mse_kernel<<<blocks, threads>>>(A, B, C, size);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

}
