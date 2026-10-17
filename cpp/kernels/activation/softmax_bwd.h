#pragma once

#include <cuda_runtime_api.h>
#include <cuda/cmath>

#include "utils.h"

void launch_softmax_bwd_kernel(float* output, float* out_grad, float* inp_grad, int prefix_dim, int reduce_dim, int suffix_dim);
