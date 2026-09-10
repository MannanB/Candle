#include "autograd.h"

MatMulGradFn::MatMulGradFn(Tensor left, Tensor right) {
    parents = {left, right};
}

std::vector<Tensor> MatMulGradFn::backward(const Tensor& output_gradient) {
    // TODO: add tranpose flag to matmul
    Tensor dLdLeft = output_gradient.matmul(parents[1].transpose(1, 2)).sum(0);
    Tensor dLdRight = parents[0].transpose(0, 1).matmul(output_gradient);
    return {dLdLeft, dLdRight};
}

AddGradFn::AddGradFn(Tensor left, Tensor right) {
    parents = {left, right};
}

std::vector<Tensor> AddGradFn::backward(const Tensor& output_gradient) {    
    Tensor left_gradient = output_gradient;
    Tensor right_gradient = output_gradient;

    while (left_gradient.ndim > parents[0].ndim) {
        left_gradient = left_gradient.sum(0);
    }
    while (right_gradient.ndim > parents[1].ndim) {
        right_gradient = right_gradient.sum(0);
    }

    return {left_gradient, right_gradient};
}
