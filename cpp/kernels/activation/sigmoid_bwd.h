#pragma once
#include <cuda_runtime_api.h>
#include <cuda/cmath>
#include "utils.h"

void launch_sigmoid_bwd_kernel(float* inp, float* out_grad, float* inp_grad, int size);
