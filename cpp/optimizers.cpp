#include "optimizers.h"
#include "kernels/tensor/add.h"

void Optimizer::add_parameter(Tensor param) {
    parameters.push_back(param);
}

void Optimizer::zero_grad() {
    for (Tensor param: parameters) {
        param.zero_grad();
    }
}

SGD::SGD(float lr) : lr(lr) {}

void SGD::step() {
    for (Tensor param: parameters) {
        launch_vec_add_kernel(
            param.tensor_data->data,
            param.grad->tensor_data->data,
            param.tensor_data->data,
            1.0,
            -lr,
            param.tensor_data->size
        );
    }
}