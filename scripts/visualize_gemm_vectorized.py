#!/usr/bin/env python3
"""Animate vectorized GEMM: float4 loads, transposed shared A and float4 stores."""

from gemm_visualization import Demo, run


if __name__ == "__main__":
    run(
        Demo(
            kind="vectorized",
            title="Vectorized GEMM",
            kernel="gemm_vectorized",
            block=16,
            depth=4,
            micro=4,
        )
    )
