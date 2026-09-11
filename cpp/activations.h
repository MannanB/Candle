#pragma once

#include "tensor.h"

struct Activations {
    static Tensor relu(const Tensor& input);
    static Tensor softmax(const Tensor& input, int dim);
};
