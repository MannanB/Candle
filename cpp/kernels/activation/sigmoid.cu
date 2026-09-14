#include "sigmoid.h"
#include <cmath>

__global__ void sigmoid_kernel(float* inp, float* out, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        float x = inp[index];
        float e = expf(-fabsf(x));
        out[index] = x >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
    }
}

void launch_sigmoid_kernel(float* inp, float* out, int size) {
    int threads = 256;
    int blocks = cuda::ceil_div(size, threads);
    sigmoid_kernel<<<blocks, threads>>>(inp, out, size);
    CUDA_CHECK(cudaGetLastError());
}
