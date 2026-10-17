#include "cce.h"

__global__ void cce_kernel(float* pred, float* real, float* out, int prefix_dim, int reduce_dim, int suffix_dim) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int count = prefix_dim * suffix_dim;
    if (index >= count) {
        return;
    }

    int prefix = index / suffix_dim;
    int suffix = index % suffix_dim;
    int offset = prefix * reduce_dim * suffix_dim + suffix;
    float loss = 0.0f;
    for (int reduce = 0; reduce < reduce_dim; ++reduce) {
        int i = offset + reduce * suffix_dim;
        loss -= real[i] * fmaxf(logf(pred[i]), -100.0f);
    }
    atomicAdd(out, loss / count);
}

void launch_cce_kernel(float* pred, float* real, float* out, int prefix_dim, int reduce_dim, int suffix_dim) {
    CUDA_CHECK(cudaMemset(out, 0, sizeof(float)));
    int threads = 256;
    int blocks = cuda::ceil_div(prefix_dim * suffix_dim, threads);
    cce_kernel<<<blocks, threads>>>(pred, real, out, prefix_dim, reduce_dim, suffix_dim);
    CUDA_CHECK(cudaGetLastError());
}
