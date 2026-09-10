#pragma once
#include "tensor.h"

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
    AddGradFn(Tensor left, Tensor right);
    std::vector<Tensor> backward(const Tensor& output_gradient);
};