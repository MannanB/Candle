#include "softmax.h"



__global__ void softmax_kernel(float* A, float* C, int prefix_dim, int reduce_dim, int suffix_dim) {
    const int out = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_count = prefix_dim * suffix_dim;

    if (out >= output_count) {
        return;
    }

    const int prefix = out / suffix_dim;
    const int suffix = out % suffix_dim;
    const int offset = prefix * reduce_dim * suffix_dim + suffix;

    float maximum = A[offset];
    for (int reduce = 1; reduce < reduce_dim; ++reduce) {
        const float value = A[offset + reduce * suffix_dim];
        maximum = value > maximum ? value : maximum;
    }

    float denominator = 0.0f;
    for (int reduce = 0; reduce < reduce_dim; ++reduce) {
        denominator += expf(A[offset + reduce * suffix_dim] - maximum);
    }

    for (int reduce = 0; reduce < reduce_dim; ++reduce) {
        const int index = offset + reduce * suffix_dim;
        C[index] = expf(A[index] - maximum) / denominator;
    }
}

void launch_softmax_kernel(float* A, float* C, int prefix_dim, int reduce_dim, int suffix_dim) {
    // assumes A is already on device
    int threads = 256;
    int blocks = cuda::ceil_div(prefix_dim*suffix_dim, threads);

    softmax_kernel<<<blocks, threads>>>(A, C, prefix_dim, reduce_dim, suffix_dim);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

}
