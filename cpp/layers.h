#pragma once
#include "tensor.h"

class Linear {
    Tensor weights;
    Tensor bias;
    bool use_bias;

public:
    Linear(int input_dim, int output_dim, bool use_bias=true);

    Tensor forward(const Tensor& input, bool grad=true);
    Tensor backward(const Tensor& dLdOutput, const Tensor& input);
};
