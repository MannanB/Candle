#pragma once
#include <cuda_runtime_api.h>
#include <cuda/cmath>
#include "utils.h"

void launch_sigmoid_kernel(float* inp, float* out, int size);
