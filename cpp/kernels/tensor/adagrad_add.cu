#include "adagrad_add.h"



__global__ void adagrad_add_kernel(float* W, float* grad, float* vel, float lr, float eps, int vectorLength) {
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;

    if(workIndex < vectorLength)
    {

        W[workIndex] = W[workIndex] - lr / (eps + sqrtf(vel[workIndex])) * grad[workIndex];
        
    }
}

void launch_adagrad_add_kernel(float* W, float* grad, float* vel, float lr, float eps, int size) {
    // assumes A and B are already on device
    int threads = 256;
    int blocks = cuda::ceil_div(size, threads);

    adagrad_add_kernel<<<blocks, threads>>>(W, grad, vel, lr, eps, size);

    CUDA_CHECK(cudaGetLastError());

}

