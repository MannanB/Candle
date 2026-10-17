#pragma once

#include "tensor.h"
#include "layers.h"

struct Optimizer {
    std::vector<Tensor*> parameters;

    virtual void add_parameter(Tensor& param);
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

    std::vector<std::shared_ptr<Tensor>> velocities;

    SGDMomentum(float lr, float momentum);
    void add_parameter(Tensor& param) override;

    void step();
};

struct AdaGrad : Optimizer {
    float lr;
    float eps;

    std::vector<std::shared_ptr<Tensor>> velocities;

    AdaGrad(float lr, float eps=0.0001);
    void add_parameter(Tensor& param) override;

    void step();
};

struct RMSProp : Optimizer {
    float lr;
    float beta;
    float eps;

    std::vector<std::shared_ptr<Tensor>> velocities;

    RMSProp(float lr, float beta, float eps=0.0001);
    void add_parameter(Tensor& param) override;

    void step();
};

struct Adam : Optimizer {
    float lr;
    float beta1;
    float beta2;
    float eps;
    int steps;

    std::vector<std::shared_ptr<Tensor>> velocities;
    std::vector<std::shared_ptr<Tensor>> momentums;

    Adam(float lr, float beta1, float beta2, float eps=0.0001);
    void add_parameter(Tensor& param) override;

    void step();
};
