#include "relu_bwd.h"



__global__ void relu_bwd_kernel(float* inp, float* out_grad, float* inp_grad, int vectorLength) {
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;

    if(workIndex < vectorLength)
    {
        inp_grad[workIndex] = out_grad[workIndex] * (inp[workIndex] > 0.0f ? 1.0f : 0.0f);
    }
}

void launch_relu_bwd_kernel(float* inp, float* out_grad, float* inp_grad, int size) {
    // assumes A is already on device
    int threads = 256;
    int blocks = cuda::ceil_div(size, threads);

    relu_bwd_kernel<<<blocks, threads>>>(inp, out_grad, inp_grad, size);

    CUDA_CHECK(cudaGetLastError());

}
