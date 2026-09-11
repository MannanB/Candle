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

AddGradFn::AddGradFn(Tensor left, Tensor right, float left_factor, float right_factor) : left_factor(left_factor), right_factor(right_factor) {
    parents = {left, right};
}

std::vector<Tensor> AddGradFn::backward(const Tensor& output_gradient) {    
    Tensor left_gradient = output_gradient * left_factor;
    Tensor right_gradient = output_gradient * right_factor;

    while (left_gradient.ndim > parents[0].ndim) {
        left_gradient = left_gradient.sum(0);
    }
    while (right_gradient.ndim > parents[1].ndim) {
        right_gradient = right_gradient.sum(0);
    }

    return {left_gradient, right_gradient};
}

ReluGradFn::ReluGradFn(Tensor input) {
    parents = {input};
}

std::vector<Tensor> ReluGradFn::backward(const Tensor& output_gradient) {
    int* out_shape = new int[output_gradient.ndim];
    for (int d = 0; d < output_gradient.ndim; ++d) {
        out_shape[d] = output_gradient.shape[d];
    }

    Tensor out(output_gradient.tensor_data->size, out_shape, output_gradient.ndim);
    launch_relu_bwd_kernel(parents[0].tensor_data->data, output_gradient.tensor_data->data, out.tensor_data->data, out.tensor_data->size);
    return {out};
}

SoftmaxGradFn::SoftmaxGradFn(Tensor input) {
    parents = {input};
}

std::vector<Tensor> SoftmaxGradFn::backward(const Tensor& output_gradient) {
    // TODO: implement? or maybe just fuse it with ce
    return {}; 
}

MSEGradFn::MSEGradFn(Tensor pred, Tensor real, int N) : N(N) {
    parents = {pred, real};
}

std::vector<Tensor> MSEGradFn::backward(const Tensor& output_gradient) {
    int* pred_shape = new int[parents[0].ndim];
    int* real_shape = new int[parents[1].ndim];
    for (int d = 0; d < parents[0].ndim; ++d) {
        pred_shape[d] = parents[0].shape[d];
        real_shape[d] = parents[1].shape[d];
    }

    Tensor dLdPred(N, pred_shape, parents[0].ndim);
    Tensor dLdReal(N, real_shape, parents[1].ndim);

    launch_mse_bwd_kernel(
        parents[0].tensor_data->data,
        parents[1].tensor_data->data,
        output_gradient.tensor_data->data,
        dLdPred.tensor_data->data,
        dLdReal.tensor_data->data,
        N
    );

    return {dLdPred, dLdReal};
}

ReshapeGradFn::ReshapeGradFn(Tensor input) {
    parents = {input};
}

std::vector<Tensor> ReshapeGradFn::backward(const Tensor& output_gradient) {
    int* input_shape = new int[parents[0].ndim];
    for (int d = 0; d < parents[0].ndim; ++d) {
        input_shape[d] = parents[0].shape[d];
    }

    Tensor input_gradient(
        output_gradient.tensor_data,
        input_shape,
        parents[0].ndim
    );
    return {input_gradient};
}
