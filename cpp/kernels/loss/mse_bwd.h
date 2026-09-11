#pragma once

#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cuda/cmath>


#include "utils.h"

void launch_mse_bwd_kernel(float* pred, float* real, float* out_grad, float* pred_grad, float* real_grad, int size);
