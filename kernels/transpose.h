#pragma once

#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cuda/cmath>


#include "utils.h"

void launch_mat_transpose_kernel(float* A, float* C, int numPrefix, int numRows, int numCols, int numSuffix);