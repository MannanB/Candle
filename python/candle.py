import os

if os.name == "nt":
    _cuda_dll_dir = os.add_dll_directory(
        os.path.join(os.environ["CUDA_PATH"], "bin")
    )

from _candle import Linear as Linear  # type: ignore  # noqa: E402
from _candle import Tensor as Tensor  # type: ignore  # noqa: E402


if __name__ == "__main__":
    layer = Linear(5, 8)
    BATCH_SIZE = 100
    inp = Tensor.uniform([BATCH_SIZE, 5, 1])
    out = layer.forward(inp)
    print(out.shape)
