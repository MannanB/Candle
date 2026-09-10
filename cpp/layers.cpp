#include "layers.h"

Linear::Linear(int input_dim, int output_dim, bool use_bias, bool requires_grad) : use_bias(use_bias) {
    int* shape = new int[2];
    shape[0] = output_dim;
    shape[1] = input_dim;

    weights = Tensor::uniform(shape, 2, 0, 1);
    weights.requires_grad = requires_grad;

    if (use_bias) {
        int* shape2 = new int[2];
        shape2[0] = output_dim;
        shape2[1] = 1;

        bias = Tensor::uniform(shape2, 2, 0, 1);
        bias.requires_grad = requires_grad;
    }
}

Tensor Linear::forward(const Tensor& input) {
    Tensor out = weights.matmul(input);

    if (use_bias) {
        out = out + bias;
    }

    return out;
}

Tensor& Linear::get_weights() {
    return weights;
}

Tensor* Linear::get_bias() {
    return use_bias ? &bias : nullptr;
}
