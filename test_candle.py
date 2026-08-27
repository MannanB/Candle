import os
import sys

_cuda_dll_dir = os.add_dll_directory(
    os.path.join(os.environ["CUDA_PATH"], "bin")
)

sys.path.insert(0, "build")
import candle


tensor1 = candle.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
tensor2 = candle.Tensor([[1, 2, 3], [5, 5, 5], [7, 7, 7], [10, 11, 12]])
tensor5 = candle.Tensor.uniform([2,3,4,5])

print(tensor1.shape)
print("\n---addition---\n")
tensor3 = tensor1 + tensor2
print(tensor1, tensor2, tensor3)

print("\n---transposition---\n")
tensor4 = tensor1.transpose(0,1)
print(tensor4.shape)
print(tensor1)
print(tensor4)
print()
print(tensor5.shape)
tensor6 = tensor5.transpose(1,2)
print(tensor6.shape)
print()
# print(tensor5)
# print(tensor6)

print("\n---matmul---\n")
# 2d
A = candle.Tensor([[1,2],[3,4],[5,6]])
B = candle.Tensor([[3,2],[1,0]])
C = A.matmul(B)
print(C.shape)
print(C)