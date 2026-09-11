#pragma once

#include "tensor.h"

struct Optimizer {
    std::vector<Tensor> parameters;

    void add_parameter(Tensor param);

    void zero_grad();

    virtual void step();

    virtual ~Optimizer() = default;
};

struct SGD : Optimizer {
    float lr;
    SGD(float lr);
    void step();
};