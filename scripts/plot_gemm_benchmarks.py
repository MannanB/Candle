#!/usr/bin/env python3
"""Benchmark every GEMM version and plot square dimension versus TFLOP/s."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys

from autotune_gemm import PROFILER, ROOT, build_environment, build_profiler


KERNELS = {
    "Naive": "gemm_naive",
    "Shared memory": "gemm_smem",
    "2D tiled": "gemm_2d_tile",
    "Vectorized": "gemm_vectorized",
    "Latest/autotuned": "gemm",
    "cuBLAS": "cublas",
}
TIME_RE = re.compile(r"Average CUDA time:\s*([0-9.]+)\s*us")


def comma_ints(text: str) -> list[int]:
    values = [int(value) for value in text.split(",")]
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("dimensions must be positive")
    return values


def benchmark(kernel: str, size: int, args: argparse.Namespace) -> float:
    command = [
        str(PROFILER), kernel,
        "--rows", str(size), "--inner", str(size), "--cols", str(size),
        "--warmup", str(args.warmup), "--iterations", str(args.iterations),
    ]
    samples = []
    for _ in range(args.repeats):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        output = result.stdout + result.stderr
        if result.returncode:
            raise RuntimeError(f"{' '.join(command)} failed:\n{output}")
        match = TIME_RE.search(output)
        if not match:
            raise RuntimeError(f"could not parse profiler output:\n{output}")
        samples.append(float(match.group(1)))
    return statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", type=comma_ints, default=comma_ints("128,256,512,1024,2048"),
        help="comma-separated square dimensions (default: 128,256,512,1024,2048)",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "benchmark_results",
        help="directory for PNG, CSV, and JSON outputs",
    )
    parser.add_argument("--no-build", action="store_true", help="use the existing profiler binary")
    args = parser.parse_args()

    if not args.no_build:
        ok, output = build_profiler(build_environment())
        if not ok:
            raise RuntimeError(f"profiler build failed:\n{output}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required; run scripts\\setup.bat") from exc

    rows = []
    for label, kernel in KERNELS.items():
        for size in args.sizes:
            elapsed_us = benchmark(kernel, size, args)
            tflops = (2.0 * size**3) / (elapsed_us * 1.0e6)
            row = {
                "label": label,
                "kernel": kernel,
                "size": size,
                "time_us": elapsed_us,
                "tflops": tflops,
            }
            rows.append(row)
            print(f"{label:17s} {size:5d}: {elapsed_us:10.3f} us  {tflops:7.3f} TFLOP/s")

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "gemm_benchmarks.csv"
    json_path = output_dir / "gemm_benchmarks.json"
    png_path = output_dir / "gemm_benchmarks.png"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().astimezone().isoformat(),
                "matrix_shape": "N x N times N x N",
                "flop_count": "2*N^3",
                "warmup": args.warmup,
                "iterations": args.iterations,
                "repeats": args.repeats,
                "results": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(10, 6))
    for label in KERNELS:
        series = [row for row in rows if row["label"] == label]
        axis.plot(
            [row["size"] for row in series],
            [row["tflops"] for row in series],
            marker="o",
            linewidth=2,
            label=label,
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(args.sizes, labels=[str(size) for size in args.sizes])
    axis.set_xlabel("Square matrix dimension (N)")
    axis.set_ylabel("Throughput (TFLOP/s)")
    axis.set_title("Candle FP32 GEMM Performance")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(png_path, dpi=180)
    plt.close(figure)

    print(f"\nGraph: {png_path}")
    print(f"CSV:   {csv_path}")
    print(f"JSON:  {json_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
