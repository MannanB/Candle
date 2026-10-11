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

struct SGDMomentum : Optimizer {
    float lr;
    float momentum; 

    std::vector<Tensor*> velocities;

    SGDMomentum(float lr, float momentum);
    void add_parameter(Tensor& param);

    void step();
}

struct AdaGrad : Optimizer {
    float lr;
    float eps;

    std::vector<Tensor*> velocities;

    AdaGrad(float lr, float eps=0.0001);
    void add_parameter(Tensor& param);

    void step();
}

struct RMSProp : Optimizer {
    float lr;
    float beta;
    float eps;

    std::vector<Tensor*> velocities;

    AdaGrad(float lr, float beta, float eps=0.0001);
    void add_parameter(Tensor& param);

    void step();
}

struct Adam : Optimizer {
    float lr;
    float beta1;
    float beta2;
    float eps;
    int steps;

    std::vector<Tensor*> velocities;
    std::vector<Tensor*> momentums;

    AdaGrad(float lr, float beta1, float beta2);
    void add_parameter(Tensor& param);

    void step();
}