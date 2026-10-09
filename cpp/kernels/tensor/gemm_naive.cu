#include "gemm_naive.h"

namespace {

constexpr int kBlockWidth = 16;

// Deliberately simple baseline: no shared memory, one output per thread.
__global__ void mat_mul_naive_kernel(
    const float* A, const float* B, float* C,
    int A_rows, int shared_dim, int B_cols) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= A_rows || col >= B_cols) {
        return;
    }

    float value = 0.0f;
    for (int k = 0; k < shared_dim; ++k) {
        value += A[row * shared_dim + k] * B[k * B_cols + col];
    }
    C[row * B_cols + col] = value;
}

}  // namespace

void launch_mat_mul_naive_kernel(
    float* A, float* B, float* C, int A_rows, int shared_dim, int B_cols) {
    const dim3 threads(kBlockWidth, kBlockWidth);
    const dim3 blocks(
        cuda::ceil_div(B_cols, kBlockWidth), cuda::ceil_div(A_rows, kBlockWidth));
    mat_mul_naive_kernel<<<blocks, threads>>>(A, B, C, A_rows, shared_dim, B_cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
