#include "relu.h"



__global__ void relu_kernel(float* A, float* C, int vectorLength) {
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;

    if(workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] > 0.0f ? A[workIndex] : 0.0f;
    }
}

void launch_relu_kernel(float* A, float* C, int size) {
    // assumes A is already on device
    int threads = 256;
    int blocks = cuda::ceil_div(size, threads);

    relu_kernel<<<blocks, threads>>>(A, C, size);

    CUDA_CHECK(cudaGetLastError());

}
