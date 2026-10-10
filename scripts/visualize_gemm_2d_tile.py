#!/usr/bin/env python3
"""Animate 2D register-tiled GEMM: shared tiles feed per-thread outer products."""

from gemm_visualization import Demo, run


if __name__ == "__main__":
    run(
        Demo(
            kind="tiled",
            title="2D register-tiled GEMM",
            kernel="gemm_2d_tile",
            block=8,
            depth=4,
            micro=2,
        )
    )
