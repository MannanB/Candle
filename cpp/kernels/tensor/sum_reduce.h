#pragma once

#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cuda/cmath>


#include "utils.h"

void launch_sum_reduce_kernel(float* A, float* B, int prefix_dim, int reduce_dim, int suffix_dim);