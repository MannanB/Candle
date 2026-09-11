#pragma once

#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cuda/cmath>


#include "utils.h"

void launch_broadcast_vec_subtract_kernel(float* A, float* B, float* C, int A_size, int B_size, int output_size);
