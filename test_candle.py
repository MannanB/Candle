import os
import sys

_cuda_dll_dir = os.add_dll_directory(
    os.path.join(os.environ["CUDA_PATH"], "bin")
)

sys.path.insert(0, "build")
import candle


tensor1 = candle.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
tensor2 = candle.Tensor([[1, 2, 3], [5, 5, 5], [7, 7, 7], [10, 11, 12]])

print(tensor1.shape)
tensor3 = tensor1 + tensor2
print(tensor1, tensor2, tensor3)