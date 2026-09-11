#include "losses.h"

#include <stdexcept>

#include "kernels/loss/mse.h"
#include "autograd.h"

Tensor Losses::mse(const Tensor& prediction, const Tensor& target) {
    if (prediction.ndim != target.ndim) {
        throw std::invalid_argument("MSE inputs must have the same shape");
    }

    for (int d = 0; d < prediction.ndim; ++d) {
        if (prediction.shape[d] != target.shape[d]) {
            throw std::invalid_argument("MSE inputs must have the same shape");
        }
    }

    int* out_shape = new int[0];
    Tensor out(1, out_shape, 0);
    launch_mse_kernel(
        prediction.tensor_data->data,
        target.tensor_data->data,
        out.tensor_data->data,
        prediction.tensor_data->size
    );

    if (prediction.requires_grad || target.requires_grad) {
        if (prediction.requires_grad) {prediction.init_grad();}
        if (target.requires_grad) {target.init_grad();}
        out.requires_grad = true;
        out.init_grad();
        out.grad_fn = std::make_shared<MSEGradFn>(prediction, target, prediction.tensor_data->size);
    }

    return out;
}
