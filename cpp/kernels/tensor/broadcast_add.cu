#include "broadcast_add.h"



__global__ void broadcast_vec_add_kernel(float* A, float* B, float* C, float A_factor, float B_factor, int A_size, int B_size, int output_size) {
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;

    if(workIndex < output_size)
    {
        C[workIndex] = A_factor * A[workIndex % A_size] + B_factor * B[workIndex % B_size];
    }
}

void launch_broadcast_vec_add_kernel(float* A, float* B, float* C, float A_factor, float B_factor, int A_size, int B_size, int output_size) {
    // assumes A and B are already on device
    int threads = 256;
    int blocks = cuda::ceil_div(output_size, threads);

    broadcast_vec_add_kernel<<<blocks, threads>>>(A, B, C, A_factor, B_factor, A_size, B_size, output_size);

    CUDA_CHECK(cudaGetLastError());

}
