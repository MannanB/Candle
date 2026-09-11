#pragma once

#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cuda/cmath>


#include "utils.h"

void launch_relu_bwd_kernel(float* inp, float* out_grad, float* inp_grad, int size);
