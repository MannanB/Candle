#include "mse_bwd.h"



__global__ void mse_bwd_kernel(float* pred, float* real, float* out_grad, float* pred_grad, float* real_grad, int vectorLength) {
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;

    if(workIndex < vectorLength)
    {
        float gradient = out_grad[0] * (2.0f / vectorLength) * (pred[workIndex] - real[workIndex]);
        pred_grad[workIndex] = gradient;
        real_grad[workIndex] = -gradient;
    }
}

void launch_mse_bwd_kernel(float* pred, float* real, float* out_grad, float* pred_grad, float* real_grad, int size) {
    // assumes inputs are already on device
    int threads = 256;
    int blocks = cuda::ceil_div(size, threads);

    mse_bwd_kernel<<<blocks, threads>>>(pred, real, out_grad, pred_grad, real_grad, size);

    CUDA_CHECK(cudaGetLastError());

}
