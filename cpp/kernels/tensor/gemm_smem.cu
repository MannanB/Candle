#include "gemm_smem.h"

namespace {

constexpr int kTileWidth = 16;

// Classic shared-memory tiled GEMM: one thread computes one output element.
__global__ void mat_mul_smem_kernel(
    const float* A, const float* B, float* C,
    int A_rows, int shared_dim, int B_cols) {
    const int row = blockIdx.y * kTileWidth + threadIdx.y;
    const int col = blockIdx.x * kTileWidth + threadIdx.x;

    __shared__ float tile_A[kTileWidth][kTileWidth];
    __shared__ float tile_B[kTileWidth][kTileWidth];

    float value = 0.0f;
    const int num_tiles = cuda::ceil_div(shared_dim, kTileWidth);
    for (int tile = 0; tile < num_tiles; ++tile) {
        const int a_col = tile * kTileWidth + threadIdx.x;
        const int b_row = tile * kTileWidth + threadIdx.y;

        tile_A[threadIdx.y][threadIdx.x] =
            row < A_rows && a_col < shared_dim ? A[row * shared_dim + a_col] : 0.0f;
        tile_B[threadIdx.y][threadIdx.x] =
            b_row < shared_dim && col < B_cols ? B[b_row * B_cols + col] : 0.0f;

        __syncthreads();
        for (int k = 0; k < kTileWidth; ++k) {
            value += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < A_rows && col < B_cols) {
        C[row * B_cols + col] = value;
    }
}

}  // namespace

void launch_mat_mul_smem_kernel(
    float* A, float* B, float* C, int A_rows, int shared_dim, int B_cols) {
    const dim3 threads(kTileWidth, kTileWidth);
    const dim3 blocks(
        cuda::ceil_div(B_cols, kTileWidth), cuda::ceil_div(A_rows, kTileWidth));
    mat_mul_smem_kernel<<<blocks, threads>>>(A, B, C, A_rows, shared_dim, B_cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
