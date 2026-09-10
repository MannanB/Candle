#pragma once
#include "tensor.h"


class Linear {
    Tensor weights;
    Tensor bias;
    bool use_bias;
    
public:




    Linear(int input_dim, int output_dim, bool use_bias=true, bool requires_grad=true);

    Tensor forward(const Tensor& input);
    Tensor& get_weights();
    Tensor* get_bias();
};
