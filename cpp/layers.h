#pragma once
#include "tensor.h"

class Linear {
    std::unique_ptr<Tensor> weights;
    std::unique_ptr<Tensor> bias;

public:
    Linear(int input_dim, int output_dim, bool use_bias=true);

    std::unique_ptr<Tensor> forward(const Tensor* input);
    std::unique_ptr<Tensor> backward(const Tensor* dLdOutput, const Tensor* input);
};
