"""Run against the built extension: python python/numeric_test_optimizers.py."""

import numpy as np

import candle


def check_optimizer(name, register_layer):
    rng = np.random.default_rng(42)
    lr, eps = 0.03, 0.02
    beta1, beta2 = 0.8, 0.95
    constructors = {
        "SGDMomentum": lambda: candle.SGDMomentum(lr, beta1),
        "AdaGrad": lambda: candle.AdaGrad(lr, eps),
        "RMSProp": lambda: candle.RMSProp(lr, beta2, eps),
        "Adam": lambda: candle.Adam(lr, beta1, beta2, eps),
    }
    optimizer = constructors[name]()
    if register_layer:
        layer = candle.Linear(3, 2)
        parameters = [layer.weights, layer.bias]
        optimizer.add_layer(layer)
    else:
        # Include negative/zero weights and a size exceeding one CUDA block.
        parameters = [
            candle.Tensor(np.linspace(-2, 2, 513).tolist()),
            candle.Tensor([[-1, 0, 1], [2, -2, 0]]),
        ]
        for parameter in parameters:
            optimizer.add_parameter(parameter)
    optimizer.init_grad()
    expected = [parameter.numpy().astype(np.float64) for parameter in parameters]
    first = [np.zeros_like(value) for value in expected]
    second = [np.zeros_like(value) for value in expected]

    for step in range(1, 9):
        optimizer.zero_grad()
        for i, parameter in enumerate(parameters):
            gradient = rng.normal(size=expected[i].shape).astype(np.float32)
            # Exercise decay of stored history when the current gradient is zero.
            if step in (1, 4):
                gradient.fill(0)
            parameter.grad.inplace_add(candle.Tensor(gradient.tolist()))
            gradient = gradient.astype(np.float64)
            if name == "SGDMomentum":
                first[i] = beta1 * first[i] - lr * gradient
                expected[i] += first[i]
            elif name == "AdaGrad":
                second[i] += gradient**2
                expected[i] -= lr * gradient / (np.sqrt(second[i]) + eps)
            elif name == "RMSProp":
                second[i] = beta2 * second[i] + (1 - beta2) * gradient**2
                expected[i] -= lr * gradient / (np.sqrt(second[i]) + eps)
            else:
                first[i] = beta1 * first[i] + (1 - beta1) * gradient
                second[i] = beta2 * second[i] + (1 - beta2) * gradient**2
                corrected_first = first[i] / (1 - beta1**step)
                corrected_second = second[i] / (1 - beta2**step)
                expected[i] -= lr * corrected_first / (np.sqrt(corrected_second) + eps)
        optimizer.step()
        for parameter, reference in zip(parameters, expected):
            np.testing.assert_allclose(
                parameter.numpy(), reference, rtol=2e-5, atol=2e-6,
                err_msg=f"{name}: step {step}, register_layer={register_layer}",
            )
        if name == "Adam":
            assert optimizer.steps == step
    print(f"PASS {name}: {'layer' if register_layer else 'parameters'}, 8 steps")


if __name__ == "__main__":
    for optimizer_name in ("SGDMomentum", "AdaGrad", "RMSProp", "Adam"):
        for use_layer in (False, True):
            check_optimizer(optimizer_name, use_layer)
