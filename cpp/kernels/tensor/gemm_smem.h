#pragma once

#include <cuda/cmath>
#include <cuda_runtime_api.h>

#include "utils.h"

void launch_mat_mul_smem_kernel(
    float* A, float* B, float* C, int A_rows, int shared_dim, int B_cols);
