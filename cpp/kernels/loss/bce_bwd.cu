#include "bce_bwd.h"

__global__ void bce_bwd_kernel(float* pred, float* real, float* out_grad, float* pred_grad, float* real_grad, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        float p = pred[index];
        float scale = out_grad[0] / size;
        pred_grad[index] = scale * (p - real[index]) / fmaxf(p * (1.0f - p), 1e-12f);
        real_grad[index] = scale * (fmaxf(log1pf(-p), -100.0f) - fmaxf(logf(p), -100.0f));
    }
}

void launch_bce_bwd_kernel(float* pred, float* real, float* out_grad, float* pred_grad, float* real_grad, int size) {
    int threads = 256;
    int blocks = cuda::ceil_div(size, threads);
    bce_bwd_kernel<<<blocks, threads>>>(pred, real, out_grad, pred_grad, real_grad, size);
    CUDA_CHECK(cudaGetLastError());
}
