#include "activations.h"

#include <stdexcept>

#include "kernels/activation/relu.h"
#include "kernels/activation/softmax.h"
#include "autograd.h"

Tensor Activations::relu(const Tensor& input) {
    int* out_shape = new int[input.ndim];
    for (int d = 0; d < input.ndim; ++d) {
        out_shape[d] = input.shape[d];
    }

    Tensor out(input.tensor_data->size, out_shape, input.ndim);

    launch_relu_kernel(
        input.tensor_data->data,
        out.tensor_data->data,
        input.tensor_data->size
    );

    
    if (input.requires_grad) {
        input.init_grad();
        out.requires_grad = true;
        out.init_grad();
        out.grad_fn = std::make_shared<ReluGradFn>(input);
    }

    return out;
}

Tensor Activations::softmax(const Tensor& input, int dim) {
    if (dim < 0 || dim >= input.ndim) {
        throw std::invalid_argument("Softmax dimension is out of range");
    }

    int prefix_dim = 1;
    for (int d = 0; d < dim; ++d) {
        prefix_dim *= input.shape[d];
    }

    const int reduce_dim = input.shape[dim];

    int suffix_dim = 1;
    for (int d = dim + 1; d < input.ndim; ++d) {
        suffix_dim *= input.shape[d];
    }

    int* out_shape = new int[input.ndim];
    for (int d = 0; d < input.ndim; ++d) {
        out_shape[d] = input.shape[d];
    }

    Tensor out(input.tensor_data->size, out_shape, input.ndim);
    launch_softmax_kernel(
        input.tensor_data->data,
        out.tensor_data->data,
        prefix_dim,
        reduce_dim,
        suffix_dim
    );
    return out;
}
