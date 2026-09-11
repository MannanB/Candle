#include "add.h"



__global__ void vec_add_kernel(float* A, float* B, float* C, int vectorLength) {
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;

    if(workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}

__global__ void inplace_vec_add_kernel(float* A, float* B, int vectorLength) {
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;

    if(workIndex < vectorLength)
    {
        A[workIndex] += B[workIndex];
    }
}

void launch_vec_add_kernel(float* A, float* B, float* C, int size) {
    // assumes A and B are already on device
    int threads = 256;
    int blocks = cuda::ceil_div(size, threads);

    vec_add_kernel<<<blocks, threads>>>(A, B, C, size);

    CUDA_CHECK(cudaGetLastError());

}

void launch_inplace_vec_add_kernel(float* A, float* B, int size) {
    // assumes A and B are already on device
    int threads = 256;
    int blocks = cuda::ceil_div(size, threads);

    inplace_vec_add_kernel<<<blocks, threads>>>(A, B, size);

    CUDA_CHECK(cudaGetLastError());

}
