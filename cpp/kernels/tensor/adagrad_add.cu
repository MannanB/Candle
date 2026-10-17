#include "adagrad_add.h"



__global__ void adagrad_add_kernel(float* W, float* grad, float* vel, float lr, float eps, int vectorLength,
                                 float momentum_correction, float velocity_correction) {
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;

    if(workIndex < vectorLength)
    {

        const float corrected_gradient = grad[workIndex] / momentum_correction;
        const float corrected_velocity = vel[workIndex] / velocity_correction;
        W[workIndex] -= lr * corrected_gradient / (eps + sqrtf(corrected_velocity));
        
    }
}

void launch_adagrad_add_kernel(float* W, float* grad, float* vel, float lr, float eps, int size) {
    launch_adagrad_add_kernel(W, grad, vel, lr, eps, size, 1.0f, 1.0f);
}

void launch_adagrad_add_kernel(float* W, float* grad, float* vel, float lr, float eps, int size,
                              float momentum_correction, float velocity_correction) {
    // assumes A and B are already on device
    int threads = 256;
    int blocks = cuda::ceil_div(size, threads);

    adagrad_add_kernel<<<blocks, threads>>>(W, grad, vel, lr, eps, size,
                                          momentum_correction, velocity_correction);

    CUDA_CHECK(cudaGetLastError());

}

