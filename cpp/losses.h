#pragma once

#include "tensor.h"

struct Losses {
    static Tensor mse(const Tensor& prediction, const Tensor& target);
    static Tensor bce(const Tensor& prediction, const Tensor& target);
    static Tensor cce(const Tensor& prediction, const Tensor& target, int dim = 1);
};
