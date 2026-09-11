#include "add.h"



__global__ void vec_add_kernel(float* A, float* B, float* C, float A_factor, float B_factor, int vectorLength) {
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;

    if(workIndex < vectorLength)
    {
        C[workIndex] = A_factor * A[workIndex] + B_factor * B[workIndex];
    }
}

void launch_vec_add_kernel(float* A, float* B, float* C, float A_factor, float B_factor, int size) {
    // assumes A and B are already on device
    int threads = 256;
    int blocks = cuda::ceil_div(size, threads);

    vec_add_kernel<<<blocks, threads>>>(A, B, C, A_factor, B_factor, size);

    CUDA_CHECK(cudaGetLastError());

}
