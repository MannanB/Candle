#pragma once

#include <cuda_runtime_api.h>

#include <cstdio>

float random_float(float min, float max);

#define CUDA_CHECK(expression) do {                                      \
    const cudaError_t result = (expression);                              \
    if (result != cudaSuccess) {                                          \
        std::fprintf(                                                     \
            stderr,                                                       \
            "CUDA Runtime Error: %s:%i:%d = %s\n",                       \
            __FILE__,                                                     \
            __LINE__,                                                     \
            result,                                                       \
            cudaGetErrorString(result)                                    \
        );                                                               \
    }                                                                    \
} while (0)
