import os

if os.name == "nt":
    _cuda_dll_dir = os.add_dll_directory(
        os.path.join(os.environ["CUDA_PATH"], "bin")
    )

from _candle import Tensor as Tensor  # type: ignore  # noqa: E402


class Linear:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        # TODO: better initialization
        self.weights = Tensor.uniform([output_dim, input_dim])
        self.biases = Tensor.uniform([output_dim, 1])

    def forward(self, x):
        return self.weights.matmul(x) + self.biases

    def backward(self, dLdOut, inp):
        # return dLdInp



        return None


if __name__ == "__main__":
    layer = Linear(5, 8)
    BATCH_SIZE = 100
    inp = Tensor.uniform([BATCH_SIZE, 5, 1])
    out = layer.forward(inp)
    print(out.shape)
