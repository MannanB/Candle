import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "build"))

_cuda_dll_dir = os.add_dll_directory(os.path.join(os.environ["CUDA_PATH"], "bin"))

from _candle import Activations as Activations  # type: ignore  # noqa: E402
from _candle import Linear as Linear  # type: ignore  # noqa: E402
from _candle import Losses as Losses  # type: ignore  # noqa: E402
from _candle import SGD as SGD  # type: ignore  # noqa: E402
from _candle import Tensor as Tensor  # type: ignore  # noqa: E402

activations = Activations
losses = Losses


if __name__ == "__main__":
    layer = Linear(5, 8)
    BATCH_SIZE = 2
    inp = Tensor.uniform([BATCH_SIZE, 5, 1], requires_grad=True)
    out = layer.forward(inp)
    print(out)
    out.backward()
    print(out)
