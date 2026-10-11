#include "optimizers.h"
#include "kernels/tensor/add.h"
#include "kernels/tensor/adagrad_add.h"

void Optimizer::add_parameter(Tensor& param) {
    parameters.push_back(&param);
}

void Optimizer::add_layer(Layer& layer) {
    for (Tensor* param: layer.get_params()) {
        add_parameter(param);
    }
}

void Optimizer::init_grad() {
    for (Tensor* param: parameters) {
        param->init_grad();
    }
}

void Optimizer::zero_grad() {
    for (Tensor* param: parameters) {
        param->zero_grad();
    }
}

SGD::SGD(float lr) : lr(lr) {}

void SGD::step() {
    for (Tensor* param: parameters) {
        launch_vec_add_kernel(
            param->tensor_data->data,
            param->grad->tensor_data->data,
            param->tensor_data->data,
            1.0,
            -lr,
            param->tensor_data->size
        );
    }
}


SGDMomentum : SGDMomentum(float lr, float momentum) : lr(lr), momentum(momentum) {}

void SGDMomentum::add_parameter(Tensor& param) {
    parameters.push_back(&param);
    // need to push back a tensor of size param.grad
    int* shape = new int[param.ndim];
    for (int d = 0; d < param.ndim; d++) {
        shape[d] = param.shape[d];
    }
    Tensor* vel = std::make_shared<Tensor>(Tensor::zeroes(shape, param.ndim));

    velocities.push_back(vel);
}

void SGDMomentum::step() {
    for (int i=0; i<vel.size(); i++) {
        Tensor* param = parameters[i];
        Tensor* vel = velocities[i];
        launch_vec_add_kernel(
            vel->tensor_data->data,
            param->grad->tensor_data->data,
            vel->tensor_data->data,
            momentum,
            -lr,
            vel->tensor_data->size
        ); // v_t+1 = p*v_t - n*g(w_t)

        launch_adagrad_add_kernel(
            param->tensor_data->data,
            vel->tensor_data->data,
            param->tensor_data->data,
            1.0,
            1.0,
            vel->tensor_data->size
        ); // w_t+1 = w_t + v_t+1
    }
}

AdaGrad : AdaGrad(float lr, float eps) : lr(lr), eps(eps) {}

void AdaGrad::add_parameter(Tensor& param) {
    parameters.push_back(&param);
    // need to push back a tensor of size param.grad
    int* shape = new int[param.ndim];
    for (int d = 0; d < param.ndim; d++) {
        shape[d] = param.shape[d];
    }
    Tensor* vel = std::make_shared<Tensor>(Tensor::zeroes(shape, param.ndim));

    velocities.push_back(vel);
}

void AdaGrad::step() {
    for (int i=0; i<vel.size(); i++) {
        Tensor* param = parameters[i];
        Tensor* vel = velocities[i];
        launch_vec_add_kernel(
            vel->tensor_data->data,
            param->grad->tensor_data->data,
            vel->tensor_data->data,
            1.0, 
            1.0,
            true, // hacky
            vel->tensor_data->size
        ); // v_t+1 = v_t + g(w_t)^2

        launch_adagrad_add_kernel(
            param->tensor_data->data,
            param->grad->tensor_data->data,
            vel->tensor_data->data,
            lr,
            eps,
            vel->tensor_data->size
        ); // w_t+1 = w_t - lr / (eps + sqrt(vel)) * g(w_t)
    }
}

RMSProp : RMSProp(float lr, float beta, float eps) : lr(lr), beta(beta), eps(eps) {}

void RMSProp::add_parameter(Tensor& param) {
    parameters.push_back(&param);
    // need to push back a tensor of size param.grad
    int* shape = new int[param.ndim];
    for (int d = 0; d < param.ndim; d++) {
        shape[d] = param.shape[d];
    }
    Tensor* vel = std::make_shared<Tensor>(Tensor::zeroes(shape, param.ndim));

    velocities.push_back(vel);
}

void RMSProp::step() {
    for (int i=0; i<vel.size(); i++) {
        Tensor* param = parameters[i];
        Tensor* vel = velocities[i];
        launch_vec_add_kernel(
            vel->tensor_data->data,
            param->grad->tensor_data->data,
            vel->tensor_data->data,
            beta, 
            1.0 - beta,
            true, // hacky
            vel->tensor_data->size
        ); // v_t+1 = B*v_t + (1-B)*g(w_t)^2

        launch_adagrad_add_kernel(
            param->tensor_data->data,
            param->grad->tensor_data->data,
            vel->tensor_data->data,
            lr,
            eps,
            vel->tensor_data->size
        ); // w_t+1 = w_t - lr / (eps + sqrt(vel)) * g(w_t)
    }
}


Adam : Adam(float lr, float beta1, float beta2, float eps) : lr(lr), beta1(beta1), beta2(beta2), eps(eps), steps(0) {}

void Adam::add_parameter(Tensor& param) {
    parameters.push_back(&param);
    // need to push back a tensor of size param.grad
    int* shape = new int[param.ndim];
    for (int d = 0; d < param.ndim; d++) {
        shape[d] = param.shape[d];
    }
    Tensor* vel = std::make_shared<Tensor>(Tensor::zeroes(shape, param.ndim));

    int* shape2 = new int[param.ndim];
    for (int d = 0; d < param.ndim; d++) {
        shape2[d] = param.shape[d];
    }
    Tensor* moment = std::make_shared<Tensor>(Tensor::zeroes(shape2, param.ndim));

    velocities.push_back(vel);
    momentums.push_back(moment)
}

void Adam::step() {
    steps++;
    for (int i=0; i<vel.size(); i++) {
        Tensor* param = parameters[i];
        Tensor* vel = velocities[i];
        Tensor* moment = momentums[i];

        float momentum_correction = (1 - std::pow(beta2, steps)); // fact check this lol
        float velocity_correction = (1 - std::pow(beta1, steps));

        launch_vec_add_kernel(
            vel->tensor_data->data,
            param->grad->tensor_data->data,
            vel->tensor_data->data,
            beta1 / velocity_correction, // is this numerically unstable?
            (1.0 - beta1) / velocity_correction,
            true, // hacky
            vel->tensor_data->size
        ); // v_t+1 = B*v_t + (1-B)*g(w_t)^2

        launch_vec_add_kernel(
            moment->tensor_data->data,
            param->grad->tensor_data->data,
            moment->tensor_data->data,
            beta2 / momentum_correction, 
            (1.0 - beta2) / momentum_correction,
            false,
            moment->tensor_data->size
        ); // m_t+1 = B*m_t + (1-B)*g(w_t)

        launch_adagrad_add_kernel(
            param->tensor_data->data,
            mo->tensor_data->data,
            vel->tensor_data->data,
            lr,
            eps,
            vel->tensor_data->size
        ); // w_t+1 = w_t - lr / (eps + sqrt(vel)) * g(w_t)
    }
}
