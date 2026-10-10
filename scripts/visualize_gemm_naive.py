#!/usr/bin/env python3
"""Animate naive GEMM: global scalar loads and one output per thread."""

from gemm_visualization import Demo, run


if __name__ == "__main__":
    run(
        Demo(
            kind="naive",
            title="Naive GEMM",
            kernel="gemm_naive",
            block=4,
            depth=1,
            micro=1,
        )
    )
