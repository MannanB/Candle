#include "gemm.h"

#define THREADS_PER_BLOCK_MM 16
#define TILE_WIDTH 16


__global__ void broadcast_gemv_kernel(float* A, float* B, float* C, int batched_dim, int A_rows, int shared_dim) {
    // C -> A_rows x B_cols

    const int batch = blockIdx.z;

    const int A_row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    const int shared_idx = blockIdx.x * TILE_WIDTH + threadIdx.x;
    const int num_tiles = ceil((float)shared_dim / TILE_WIDTH);

    __shared__ float tile_shared_mem_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_mem_B[TILE_WIDTH];

    float value = 0;


    // load tiles to shared mem
    if (batch < batched_dim && A_row < A_rows && shared_idx < shared_dim) {
        tile_shared_mem_A[threadIdx.y][threadIdx.x] = A[A_row * shared_dim + shared_idx];
    } else {
        tile_shared_mem_A[threadIdx.y][threadIdx.x] = 0;
    }
    
    if (threadIdx.y == 0) {
        if (batch < batched_dim && shared_idx < shared_dim) {
            shared_mem_B[threadIdx.x] = B[shared_idx];
        } else {
            shared_mem_B[threadIdx.x] = 0;
        }
    }

    __syncthreads();

    for (int k_tile = 0; k_tile < blockDim.x; k_tile++) {
        value += tile_shared_mem_A[threadIdx.y][k_tile] * shared_mem_B[];
    }
    __syncthreads();

    

    if (batch < batched_dim && row < A_rows && col < B_cols) {
        C[batch * B_cols * A_rows + row*B_cols + col] = value;
    }
}

void launch_broadcast_gemv_kernel(float* A, float* B, float* C, int batched_dim, int A_rows, int shared_dim) {
    // A is m x k, B is batched_dim x k x 1 (n=1 for NNs)
    dim3 threadsPerBlock(THREADS_PER_BLOCK_MM, THREADS_PER_BLOCK_MM);
    dim3 numBlocks(cuda::ceil_div(shared_dim, TILE_WIDTH), cuda::ceil_div(A_rows, TILE_WIDTH), batched_dim);

    broadcast_gemv_kernel<<<numBlocks, threadsPerBlock>>>(A, B, C, batched_dim, A_rows, shared_dim);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

}


