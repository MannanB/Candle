#include "add.h"



__global__ void vec_add_kernel(float* A, float* B, float* C, float A_factor, float B_factor, bool square_lhs, int vectorLength) {
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;

    if(workIndex < vectorLength)
    {
        if (!square_lhs) {
            C[workIndex] = A_factor * A[workIndex] + B_factor * B[workIndex];
        } else {
            // lol just for AdaGrad/RMSprop/Adam #lazy
            C[workIndex] = A_factor * A[workIndex] + B_factor * B[workIndex] * B[workIndex];
        }
    }
}

void launch_vec_add_kernel(float* A, float* B, float* C, float A_factor, float B_factor, bool square_lhs, int size) {
    // assumes A and B are already on device
    int threads = 256;
    int blocks = cuda::ceil_div(size, threads);

    vec_add_kernel<<<blocks, threads>>>(A, B, C, A_factor, B_factor, square_lhs, size);

    CUDA_CHECK(cudaGetLastError());

}

void launch_vec_add_kernel(float* A, float* B, float* C, float A_factor, float B_factor, int size) {
    launch_vec_add_kernel(A, B, C, A_factor, B_factor, false, size);
}

