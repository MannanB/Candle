#pragma once
#include "tensor.h"


class Layer {
    // oop or something its 12:30 am
public:
    virtual std::vector<Tensor*> get_params()=0;
};

class Linear : public Layer {
    Tensor weights;
    Tensor bias;
    bool use_bias;
    
public:

    Linear(int input_dim, int output_dim, bool use_bias=true, bool requires_grad=true);

    std::vector<Tensor*> get_params();
    Tensor forward(const Tensor& input);
    Tensor& get_weights();
    Tensor* get_bias();
};
