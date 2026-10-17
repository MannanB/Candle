#pragma once

#include <cuda_runtime_api.h>
#include <cuda/cmath>

#include "utils.h"

void launch_bce_kernel(float* pred, float* real, float* out, int size);
