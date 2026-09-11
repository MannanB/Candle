#pragma once
#include "tensor.h"
#include "kernels/activation/relu_bwd.h"
#include "kernels/loss/mse_bwd.h"

struct GradFn {
    std::vector<Tensor> parents;
    virtual std::vector<Tensor> backward(const Tensor& output_gradient) = 0;

    virtual ~GradFn() = default;
};

// TODO: all these grad fns are for broadcast only
// probably shouldnt do that 

struct MatMulGradFn : GradFn {
    MatMulGradFn(Tensor left, Tensor right);
    std::vector<Tensor> backward(const Tensor& output_gradient);
};

struct AddGradFn : GradFn {
    float left_factor;
    float right_factor;
    AddGradFn(Tensor left, Tensor right, float left_factor, float right_factor);
    std::vector<Tensor> backward(const Tensor& output_gradient);
};


struct ReluGradFn : GradFn {
    ReluGradFn(Tensor input);
    std::vector<Tensor> backward(const Tensor& output_gradient);
};

struct SoftmaxGradFn : GradFn {
    SoftmaxGradFn(Tensor input);
    std::vector<Tensor> backward(const Tensor& output_gradient);
};

struct MSEGradFn : GradFn {
    int N;
    MSEGradFn(Tensor pred, Tensor real, int N);
    std::vector<Tensor> backward(const Tensor& output_gradient);
};

struct ReshapeGradFn : GradFn {
    ReshapeGradFn(Tensor input);
    std::vector<Tensor> backward(const Tensor& output_gradient);
};
