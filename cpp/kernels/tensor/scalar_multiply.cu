#include "scalar_multiply.h"



__global__ void scalar_multiply_kernel(float* A, float scalar, float* C, int vectorLength) {
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;

    if(workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] * scalar;
    }
}

void launch_scalar_multiply_kernel(float* A, float scalar, float* C, int size) {
    // assumes A is already on device
    int threads = 256;
    int blocks = cuda::ceil_div(size, threads);

    scalar_multiply_kernel<<<blocks, threads>>>(A, scalar, C, size);

    CUDA_CHECK(cudaGetLastError());

}
