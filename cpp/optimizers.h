#pragma once

#include "tensor.h"
#include "layers.h"

struct Optimizer {
    std::vector<Tensor*> parameters;

    void add_parameter(Tensor& param);
    void add_layer(Layer& layer);
    void init_grad();
    void zero_grad();

    virtual void step() = 0;

    virtual ~Optimizer() = default;
};

struct SGD : Optimizer {
    float lr;
    SGD(float lr);
    void step();
};
