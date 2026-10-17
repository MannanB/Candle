#include "softmax_bwd.h"

__global__ void softmax_bwd_kernel(float* output, float* out_grad, float* inp_grad, int prefix_dim, int reduce_dim, int suffix_dim) {
    int out = blockIdx.x * blockDim.x + threadIdx.x;
    if (out >= prefix_dim * suffix_dim) {
        return;
    }

    int prefix = out / suffix_dim;
    int suffix = out % suffix_dim;
    int offset = prefix * reduce_dim * suffix_dim + suffix;
    float dot = 0.0f;
    for (int reduce = 0; reduce < reduce_dim; ++reduce) {
        int index = offset + reduce * suffix_dim;
        dot += out_grad[index] * output[index];
    }
    for (int reduce = 0; reduce < reduce_dim; ++reduce) {
        int index = offset + reduce * suffix_dim;
        inp_grad[index] = output[index] * (out_grad[index] - dot);
    }
}

void launch_softmax_bwd_kernel(float* output, float* out_grad, float* inp_grad, int prefix_dim, int reduce_dim, int suffix_dim) {
    int threads = 256;
    int blocks = cuda::ceil_div(prefix_dim * suffix_dim, threads);
    softmax_bwd_kernel<<<blocks, threads>>>(output, out_grad, inp_grad, prefix_dim, reduce_dim, suffix_dim);
    CUDA_CHECK(cudaGetLastError());
}
