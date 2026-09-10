#include "layers.h"

Linear::Linear(int input_dim, int output_dim, bool use_bias) : use_bias(use_bias) {
    int* shape = new int[2];
    shape[0] = output_dim;
    shape[1] = input_dim;

    weights = Tensor::uniform(shape, 2, 0, 1);

    if (use_bias) {
        int* shape2 = new int[2];
        shape2[0] = output_dim;
        shape2[1] = 1;

        bias = Tensor::uniform(shape2, 2, 0, 1);
    }
}

Tensor Linear::forward(const Tensor& input, bool grad) {
    GradGraphNode node;

    if (grad) {
        std::vector<Parameter> params;
        Parameter weight_param;
        weight_param.parameter = &weights;
        params.push_back(weight_param);
        if (use_bias) {
            Parameter bias_param;
            bias_param.parameter = &bias;
            params.push_back(bias_param);
        }
        node.parameters = params;
        std::vector<GradGraphNode*> prev = {input.grad_graph};
        node.previous = prev;
        std::vector<const Tensor*> inputs = {&input};
        node.inputs = inputs;
        input.grad_graph->next.push_back(&node);
    }

    Tensor out = weights.matmul(input);



    if (use_bias) {


        out = out + bias;
    }

    if (grad) {
        out.grad_graph = &node;
    }

    return out;
}

Tensor Linear::backward(const Tensor& dLdOutput, const Tensor& input) {
    // return dLdInput, dLdW, dLdb
    // sums across batch dim
    if (use_bias) {
        Tensor dLdb = dLdOutput.sum(0);
    }

    Tensor dLdW = dLdOutput.matmul(input.transpose(1, 2)).sum(0);
    Tensor dLdInput = weights.transpose(0, 1).matmul(dLdOutput);
    return dLdInput;
}
