#include "sum_reduce.h"



__global__ void sum_reduce_kernel(float* A, float* B, int prefix_dim, int reduce_dim, int suffix_dim) {
    const int out = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_count = prefix_dim * suffix_dim;

    if (out >= output_count) {
        return;
    }

    const int prefix = out / suffix_dim;
    const int suffix = out % suffix_dim;
    const float* input = A + prefix * reduce_dim * suffix_dim + suffix;

    float accumulator = 0.0f;
    for (int reduce = 0; reduce < reduce_dim; ++reduce) {
        accumulator += input[reduce * suffix_dim];
    }

    B[out] = accumulator;
}

void launch_sum_reduce_kernel(float* A, float* B, int prefix_dim, int reduce_dim, int suffix_dim) {
    int threads = 256;
    int blocks = cuda::ceil_div(prefix_dim*reduce_dim*suffix_dim, threads);

    sum_reduce_kernel<<<blocks, threads>>>(A, B, prefix_dim, reduce_dim, suffix_dim);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


}