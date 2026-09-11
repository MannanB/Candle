#pragma once

#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>  
#include <stdio.h>
#include <cuda/cmath>


#include "utils.h"

void launch_broadcast_mat_mul_kernel(float* A, float* B, float* C, int batched_dim, int A_rows, int shared_dim, int B_cols);