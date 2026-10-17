#include "cce_bwd.h"

__global__ void cce_bwd_kernel(float* pred, float* real, float* out_grad, float* pred_grad, float* real_grad, int size, int count) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        float scale = out_grad[0] / count;
        pred_grad[index] = -scale * real[index] / fmaxf(pred[index], 1e-12f);
        real_grad[index] = -scale * fmaxf(logf(pred[index]), -100.0f);
    }
}

void launch_cce_bwd_kernel(float* pred, float* real, float* out_grad, float* pred_grad, float* real_grad, int size, int count) {
    int threads = 256;
    int blocks = cuda::ceil_div(size, threads);
    cce_bwd_kernel<<<blocks, threads>>>(pred, real, out_grad, pred_grad, real_grad, size, count);
    CUDA_CHECK(cudaGetLastError());
}
