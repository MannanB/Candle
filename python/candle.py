from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "build"))

_cuda_dll_dir = os.add_dll_directory(os.path.join(os.environ["CUDA_PATH"], "bin"))

if TYPE_CHECKING:
    # Editor-only declarations. Keep signatures aligned with cpp/bindings.cpp.
    from typing import TypeAlias, Union, overload

    import numpy as np
    from numpy.typing import NDArray

    _TensorValues: TypeAlias = Union[float, list["_TensorValues"]]

    class TensorData:
        def __init__(self) -> None: ...

        @property
        def size(self) -> int: ...

    class Tensor:
        requires_grad: bool

        @overload
        def __init__(self) -> None: ...
        @overload
        def __init__(self, data_list: list[_TensorValues], requires_grad: bool = False) -> None: ...

        @staticmethod
        def uniform(
            shape: list[int], min: float = 0.0, max: float = 1.0,
            requires_grad: bool = False,
        ) -> Tensor: ...

        @staticmethod
        def ones(shape: list[int], requires_grad: bool = False) -> Tensor: ...

        @staticmethod
        def zeroes(shape: list[int], requires_grad: bool = False) -> Tensor: ...

        @property
        def tensor_data(self) -> TensorData: ...

        @property
        def shape(self) -> list[int]: ...

        @property
        def grad(self) -> Tensor | None: ...

        @property
        def is_leaf(self) -> bool: ...

        def numpy(self) -> NDArray[np.float32]: ...
        def tolist(self) -> _TensorValues: ...
        def save(self, path: str) -> bool: ...

        @staticmethod
        @overload
        def load(path: str) -> Tensor: ...
        @staticmethod
        @overload
        def load(path: str, tensor_data: TensorData) -> Tensor: ...

        @staticmethod
        def save_many(tensors: list[Tensor], path: str) -> bool: ...

        @staticmethod
        @overload
        def load_many(path: str) -> list[Tensor]: ...
        @staticmethod
        @overload
        def load_many(path: str, tensors: list[Tensor]) -> int: ...

        def backward(self) -> None: ...
        def add(
            self, other: Tensor, self_factor: float = 1.0, other_factor: float = 1.0,
        ) -> Tensor: ...
        def inplace_add(
            self, other: Tensor, self_factor: float = 1.0, other_factor: float = 1.0,
        ) -> None: ...
        def __add__(self, other: Tensor) -> Tensor: ...
        def __sub__(self, other: Tensor) -> Tensor: ...
        def __mul__(self, other: float) -> Tensor: ...
        def transpose(self, dim1: int, dim2: int) -> Tensor: ...
        def sum(self, dim: int) -> Tensor: ...

        @overload
        def flatten(self) -> Tensor: ...
        @overload
        def flatten(self, dim1: int, dim2: int) -> Tensor: ...

        def unsqueeze(self) -> Tensor: ...
        def matmul(
            self, other: Tensor, transpose_self: bool = False, transpose_other: bool = False,
        ) -> Tensor: ...
        def __repr__(self) -> str: ...

    class Activations:
        @staticmethod
        def relu(input: Tensor) -> Tensor: ...

        @staticmethod
        def sigmoid(input: Tensor) -> Tensor: ...

        @staticmethod
        def softmax(input: Tensor, dim: int) -> Tensor: ...

    class Losses:
        @staticmethod
        def mse(prediction: Tensor, target: Tensor) -> Tensor: ...

        @staticmethod
        def bce(prediction: Tensor, target: Tensor) -> Tensor:
            """Mean binary cross-entropy; prediction contains probabilities."""
            ...

        @staticmethod
        def cce(prediction: Tensor, target: Tensor, dim: int = 1) -> Tensor:
            """Mean categorical cross-entropy; probabilities and one-hot/soft targets."""
            ...

    class _Layer: ...

    class Linear(_Layer):
        def __init__(
            self, input_dim: int, output_dim: int, use_bias: bool = True,
            requires_grad: bool = True,
        ) -> None: ...

        @property
        def weights(self) -> Tensor: ...

        @property
        def bias(self) -> Tensor | None: ...

        def forward(self, input: Tensor) -> Tensor: ...
        def __call__(self, input: Tensor) -> Tensor: ...
        def save(self, path: str) -> bool: ...

        @staticmethod
        def load(path: str) -> Linear: ...

    class _Optimizer:
        lr: float

        def add_parameter(self, parameter: Tensor) -> None: ...
        def add_layer(self, layer: _Layer) -> None: ...
        def init_grad(self) -> None: ...
        def zero_grad(self) -> None: ...
        def step(self) -> None: ...

    class SGD(_Optimizer):
        def __init__(self, lr: float) -> None: ...

    class SGDMomentum(_Optimizer):
        momentum: float

        def __init__(self, lr: float, momentum: float) -> None: ...

    class AdaGrad(_Optimizer):
        eps: float

        def __init__(self, lr: float, eps: float = 0.0001) -> None: ...

    class RMSProp(_Optimizer):
        beta: float
        eps: float

        def __init__(self, lr: float, beta: float, eps: float = 0.0001) -> None: ...

    class Adam(_Optimizer):
        beta1: float
        beta2: float
        eps: float

        def __init__(
            self, lr: float, beta1: float, beta2: float, eps: float = 0.0001,
        ) -> None: ...

        @property
        def steps(self) -> int: ...
else:
    from _candle import Activations as Activations  # type: ignore  # noqa: E402
    from _candle import AdaGrad as AdaGrad  # type: ignore  # noqa: E402
    from _candle import Adam as Adam  # type: ignore  # noqa: E402
    from _candle import Linear as Linear  # type: ignore  # noqa: E402
    from _candle import Losses as Losses  # type: ignore  # noqa: E402
    from _candle import RMSProp as RMSProp  # type: ignore  # noqa: E402
    from _candle import SGD as SGD  # type: ignore  # noqa: E402
    from _candle import SGDMomentum as SGDMomentum  # type: ignore  # noqa: E402
    from _candle import Tensor as Tensor  # type: ignore  # noqa: E402
    from _candle import TensorData as TensorData  # type: ignore  # noqa: E402

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
