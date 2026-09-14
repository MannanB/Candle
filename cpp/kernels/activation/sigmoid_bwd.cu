#include "sigmoid_bwd.h"
#include <cmath>

__global__ void sigmoid_bwd_kernel(float* inp, float* out_grad, float* inp_grad, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        float e = expf(-fabsf(inp[index]));
        float denominator = 1.0f + e;
        inp_grad[index] = out_grad[index] * e / (denominator * denominator);
    }
}

void launch_sigmoid_bwd_kernel(float* inp, float* out_grad, float* inp_grad, int size) {
    int threads = 256;
    int blocks = cuda::ceil_div(size, threads);
    sigmoid_bwd_kernel<<<blocks, threads>>>(inp, out_grad, inp_grad, size);
    CUDA_CHECK(cudaGetLastError());
}
