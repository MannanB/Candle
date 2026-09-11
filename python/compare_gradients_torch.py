import numpy as np
import torch

from candle import Linear, Tensor, activations, losses


def compare(name, candle_value, torch_value, rtol=1e-3, atol=1e-3):
    candle_array = np.asarray(candle_value)
    torch_array = torch_value.detach().cpu().numpy()
    matches = np.allclose(candle_array, torch_array, rtol=rtol, atol=atol)
    max_error = np.max(np.abs(candle_array - torch_array))

    print(f"{name}: {'PASS' if matches else 'FAIL'} (max error: {max_error})")
    if not matches:
        print("Candle:")
        print(candle_array)
        print("PyTorch:")
        print(torch_array)

    return matches


def main():
    rng = np.random.default_rng(0)

    batch_size = 4
    input_dim = 5
    output_dim = 3

    input_data = rng.normal(size=(batch_size, input_dim, 1)).astype(np.float32)
    target_data = rng.normal(size=(batch_size, output_dim, 1)).astype(np.float32)

    candle_input = Tensor(input_data.tolist(), requires_grad=True)
    candle_target = Tensor(target_data.tolist())
    candle_linear = Linear(input_dim, output_dim, use_bias=True, requires_grad=True)

    torch_input = torch.tensor(
        input_data.squeeze(-1),
        dtype=torch.float32,
        requires_grad=True,
    )
    torch_target = torch.tensor(target_data.squeeze(-1), dtype=torch.float32)
    torch_linear = torch.nn.Linear(input_dim, output_dim, bias=True)

    with torch.no_grad():
        torch_linear.weight.copy_(
            torch.from_numpy(candle_linear.weights.numpy())
        )
        torch_linear.bias.copy_(
            torch.from_numpy(candle_linear.bias.numpy().reshape(-1))
        )

    candle_linear_output = candle_linear(candle_input)
    candle_relu_output = activations.relu(candle_linear_output)
    candle_loss = losses.mse(candle_relu_output, candle_target)
    candle_loss.backward()

    torch_linear_output = torch_linear(torch_input)
    torch_relu_output = torch.relu(torch_linear_output)
    torch_loss = torch.nn.functional.mse_loss(
        torch_relu_output,
        torch_target,
        reduction="mean",
    )
    torch_loss.backward()

    results = [
        compare(
            "linear output",
            candle_linear_output.numpy().squeeze(-1),
            torch_linear_output,
        ),
        compare(
            "relu output",
            candle_relu_output.numpy().squeeze(-1),
            torch_relu_output,
        ),
        compare("mse loss", candle_loss.numpy(), torch_loss),
        compare(
            "input gradient",
            candle_input.grad.numpy().squeeze(-1),
            torch_input.grad,
        ),
        compare(
            "weight gradient",
            candle_linear.weights.grad.numpy(),
            torch_linear.weight.grad,
        ),
        compare(
            "bias gradient",
            candle_linear.bias.grad.numpy().reshape(-1),
            torch_linear.bias.grad,
        ),
    ]

    if not all(results):
        raise AssertionError("Candle gradients do not match PyTorch")

    print("All forward and backward comparisons passed.")


if __name__ == "__main__":
    main()
