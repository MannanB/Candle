#include "broadcast_add.h"



__global__ void vec_add_kernel(float* A, float* B, float* C, int vectorLength, int batch_size) {
    const int batch = blockIdx.y;

    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;

    if(workIndex < vectorLength)
    {
        C[batch * vectorLength + workIndex] = A[batch * vectorLength + workIndex] + B[workIndex];
    }
}

void launch_broadcast_vec_add_kernel(float* A, float* B, float* C, int size, int batch_dim) {
    // assumes A and B are already on device
    // B is batch_size x size
    int threads = 256;

    dim3 numBlocks(cuda::ceil_div(size, threads), batch_dim);


    vec_add_kernel<<<numBlocks, threads>>>(A, B, C, size, batch_dim);

    CUDA_CHECK(cudaGetLastError());

}
