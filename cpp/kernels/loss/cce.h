#pragma once

#include <cuda_runtime_api.h>
#include <cuda/cmath>

#include "utils.h"

void launch_cce_kernel(float* pred, float* real, float* out, int prefix_dim, int reduce_dim, int suffix_dim);
