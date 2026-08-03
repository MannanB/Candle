import os
import sys

_cuda_dll_dir = os.add_dll_directory(
    os.path.join(os.environ["CUDA_PATH"], "bin")
)

sys.path.insert(0, "build")
import candle

print(candle.add(1, 2))