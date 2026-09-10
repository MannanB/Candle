#include "layers.h"

Linear::Linear(int input_dim, int output_dim, bool use_bias) {
    int* shape = new int[2];
    shape[0] = output_dim;
    shape[1] = input_dim;

    weights = Tensor::uniform(shape, 2, 0, 1);

    if (use_bias) {
        int* shape2 = new int[2];
        shape2[0] = output_dim;
        shape2[1] = 1;

        bias = Tensor::uniform(shape2, 2, 0, 1);
    } else {
        bias = nullptr;
    }
}

std::unique_ptr<Tensor> Linear::forward(const Tensor* input) {
    std::unique_ptr<Tensor> out = weights->matmul(*input);
    if (bias != nullptr) {
        out = *out + *bias;
    }
    return out;
}

std::unique_ptr<Tensor> Linear::backward(const Tensor* dLdOutput, const Tensor* input) {
    // return dLdInput, dLdW, dLdb
    // sums across batch dim

    std::unique_ptr<Tensor> dLdb = dLdOutput->sum(0);
    std::unique_ptr<Tensor> dLdW = dLdOutput->matmul(*(input->transpose(1, 2)))->sum(0);
    std::unique_ptr<Tensor> dLdInput = weights->transpose(0, 1)->matmul(*dLdOutput);
}