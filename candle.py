import sys, os

_cuda_dll_dir = os.add_dll_directory(
    os.path.join(os.environ["CUDA_PATH"], "bin")
)

sys.path.insert(0, "build")
from candle import *

class Linear:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        # TODO: better initialization
        self.weights = Tensor.uniform([output_dim, input_dim])
        self.biases = Tensor.uniform([output_dim])
 
    def forward(self, x): 
        return self.weights.matmul(x) + self.biases

    def backward(self, y):
        return None