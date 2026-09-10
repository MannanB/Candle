#include "autograd.h"

MatMulGradFn::MatMulGradFn(Tensor right, Tensor left) {
    parents = {right, left};
}

std::vector<Tensor> MatMulGradFn::backward(const Tensor& output_gradient) {
    // TODO: add tranpose flag to matmul
    Tensor dLdLeft = output_gradient.matmul(parents[0].transpose(1, 2)).sum(0);
    Tensor dLdRight = parents[1].transpose(0, 1).matmul(output_gradient);
    return {dLdRight, dLdLeft};
}

AddGradFn::AddGradFn(Tensor right, Tensor left) {
    parents = {right, left};
}

std::vector<Tensor> AddGradFn::backward(const Tensor& output_gradient) {    
    return {output_gradient, output_gradient}; // left and right have same gradients
}