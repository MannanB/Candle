#!/usr/bin/env python3
"""Animate shared-memory GEMM: whole-tile loads and one output per thread."""

from gemm_visualization import Demo, run


if __name__ == "__main__":
    run(
        Demo(
            kind="smem",
            title="Shared-memory GEMM",
            kernel="gemm_smem",
            block=4,
            depth=4,
            micro=1,
        )
    )
