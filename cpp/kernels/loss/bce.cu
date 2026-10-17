#include "bce.h"

__global__ void bce_kernel(float* pred, float* real, float* out, int size) {
    __shared__ float partialSums[256];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    float loss = 0.0f;

    if (index < size) {
        float p = pred[index];
        float y = real[index];
        loss = -y * fmaxf(logf(p), -100.0f) - (1.0f - y) * fmaxf(log1pf(-p), -100.0f);
    }
    partialSums[threadIdx.x] = loss;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partialSums[threadIdx.x] += partialSums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(out, partialSums[0] / size);
    }
}

void launch_bce_kernel(float* pred, float* real, float* out, int size) {
    CUDA_CHECK(cudaMemset(out, 0, sizeof(float)));
    int threads = 256;
    int blocks = cuda::ceil_div(size, threads);
    bce_kernel<<<blocks, threads>>>(pred, real, out, size);
    CUDA_CHECK(cudaGetLastError());
}
