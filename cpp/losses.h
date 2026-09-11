#pragma once

#include "tensor.h"

struct Losses {
    static Tensor mse(const Tensor& prediction, const Tensor& target);
};
