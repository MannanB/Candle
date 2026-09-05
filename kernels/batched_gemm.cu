#include "gemm.h"

#define THREADS_PER_BLOCK_MM 16
#define TILE_WIDTH 16


__global__ void batched_mat_mul_kernel(float* A, float* B, float* C, int batched_dim, int A_rows, int shared_dim, int B_cols) {
    // C -> A_rows x B_cols

    const int batch = blockIdx.z;
    const int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    const int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    const int num_tiles = ceil((float)shared_dim / TILE_WIDTH);

    __shared__ float tile_shared_mem_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_shared_mem_B[TILE_WIDTH][TILE_WIDTH];

    float value = 0;

    for (int tile = 0; tile < num_tiles; tile++) {
        // load tiles to shared mem
        if (batch < batched_dim && row < A_rows && (tile * blockDim.x + threadIdx.x) < shared_dim) {
            tile_shared_mem_A[threadIdx.y][threadIdx.x] = A[batch * shared_dim * A_rows + row * shared_dim + (tile * blockDim.x + threadIdx.x)];
        } else {
            tile_shared_mem_A[threadIdx.y][threadIdx.x] = 0;
        }
        
        if (batch < batched_dim && (tile * blockDim.y + threadIdx.y) < shared_dim && col < B_cols) {
            tile_shared_mem_B[threadIdx.y][threadIdx.x] = B[batch * shared_dim * B_cols + (tile * blockDim.y + threadIdx.y) * B_cols + col];
        } else {
            tile_shared_mem_B[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        for (int k_tile = 0; k_tile < blockDim.x; k_tile++) {
            value += tile_shared_mem_A[threadIdx.y][k_tile] * tile_shared_mem_B[k_tile][threadIdx.x];
        }
        __syncthreads();

    }

    if (batch < batched_dim && row < A_rows && col < B_cols) {
        C[batch * B_cols * A_rows + row*B_cols + col] = value;
    }
}

void launch_batched_mat_mul_kernel(float* A, float* B, float* C, int batched_dim, int A_rows, int shared_dim, int B_cols) {

    dim3 threadsPerBlock(THREADS_PER_BLOCK_MM, THREADS_PER_BLOCK_MM);
    dim3 numBlocks(cuda::ceil_div(B_cols, TILE_WIDTH), cuda::ceil_div(A_rows, TILE_WIDTH), batched_dim);

    batched_mat_mul_kernel<<<numBlocks, threadsPerBlock>>>(A, B, C, batched_dim, A_rows, shared_dim, B_cols);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

}


