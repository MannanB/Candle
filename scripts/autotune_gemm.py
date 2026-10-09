#!/usr/bin/env python3
"""Brute-force the compile-time tile constants in the current GEMM kernel."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "cpp" / "kernels" / "tensor" / "gemm.cu"
BUILD_DIR = ROOT / "build"
PROFILER = BUILD_DIR / ("CandleProfiler.exe" if os.name == "nt" else "CandleProfiler")
TIME_RE = re.compile(r"Average CUDA time:\s*([0-9.]+)\s*us")


def csv_ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",")]


def parse_shapes(text: str) -> list[tuple[int, int, int]]:
    shapes: list[tuple[int, int, int]] = []
    for item in text.split(","):
        dims = [int(value) for value in item.lower().split("x")]
        if len(dims) == 1:
            dims *= 3
        if len(dims) != 3 or any(value <= 0 for value in dims):
            raise argparse.ArgumentTypeError(
                f"invalid shape '{item}'; use N or MxKxN"
            )
        shapes.append((dims[0], dims[1], dims[2]))
    return shapes


def candidate_configs(args: argparse.Namespace) -> list[dict[str, int]]:
    candidates = []
    for bm, bn, bk, tm, tn in itertools.product(
        args.bm, args.bn, args.bk, args.tm, args.tn
    ):
        if bm % tm or bn % tn or bk % 4 or tn % 4:
            continue

        threads = (bm // tm) * (bn // tn)
        if threads < 32 or threads > 1024 or threads % 32:
            continue

        # Strict current design: every thread loads exactly one float4
        # from each input tile.
        if bm * bk != 4 * threads or bn * bk != 4 * threads:
            continue

        candidates.append(
            {"BM": bm, "BN": bn, "BK": bk, "TM": tm, "TN": tn,
             "threads": threads}
        )
    return candidates


def apply_config(source: str, config: dict[str, int]) -> str:
    updated = source
    for name in ("BM", "BN", "BK", "TM", "TN"):
        pattern = re.compile(rf"^(#define\s+{name}\s+)\d+", re.MULTILINE)
        updated, count = pattern.subn(rf"\g<1>{config[name]}", updated, count=1)
        if count != 1:
            raise RuntimeError(f"could not find exactly one #define for {name}")
    return updated


def cmake_command() -> Path:
    cache = BUILD_DIR / "CMakeCache.txt"
    if cache.exists():
        for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("CMAKE_COMMAND:INTERNAL="):
                command = Path(line.split("=", 1)[1])
                if command.exists():
                    return command
    return Path("cmake")


def build_environment() -> dict[str, str]:
    env = os.environ.copy()
    if os.name != "nt" or env.get("VSCMD_VER"):
        return env

    vs_root = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    vcvars = vs_root / "Microsoft Visual Studio" / "2022" / "BuildTools" / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if not vcvars.exists():
        return env

    result = subprocess.run(
        f'call "{vcvars}" >nul && set',
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            env[key] = value
    return env


def build_profiler(env: dict[str, str]) -> tuple[bool, str]:
    result = subprocess.run(
        [str(cmake_command()), "--build", str(BUILD_DIR), "--target", "candle_profiler", "-j", str(os.cpu_count() or 4)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    return result.returncode == 0, output


def benchmark(shape: tuple[int, int, int], args: argparse.Namespace) -> float:
    m, k, n = shape
    command = [
        str(PROFILER), "gemm",
        "--rows", str(m), "--inner", str(k), "--cols", str(n),
        "--warmup", str(args.warmup), "--iterations", str(args.iterations),
    ]
    samples = []
    for _ in range(args.repeats):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        output = result.stdout + result.stderr
        if result.returncode:
            raise RuntimeError(output.strip())
        match = TIME_RE.search(output)
        if not match:
            raise RuntimeError(f"could not parse profiler output:\n{output}")
        samples.append(float(match.group(1)))
    return statistics.median(samples)


def shape_is_compatible(shape: tuple[int, int, int], config: dict[str, int]) -> bool:
    m, k, n = shape
    return m % config["BM"] == 0 and n % config["BN"] == 0 and k % config["BK"] == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=parse_shapes, default=parse_shapes("64,128,512,1024,2048"),
                        help="comma-separated N or MxKxN shapes")
    parser.add_argument("--bm", type=csv_ints, default=csv_ints("32,64,128"))
    parser.add_argument("--bn", type=csv_ints, default=csv_ints("32,64,128"))
    parser.add_argument("--bk", type=csv_ints, default=csv_ints("4,8,16,32"))
    parser.add_argument("--tm", type=csv_ints, default=csv_ints("2,4,8,16"))
    parser.add_argument("--tn", type=csv_ints, default=csv_ints("4,8,16"))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "autotune_results" / "gemm_autotune.json")
    args = parser.parse_args()

    candidates = candidate_configs(args)
    if not candidates:
        parser.error("no configurations satisfy the strict float4 constraints")
    if not SOURCE.exists() or not (BUILD_DIR / "CMakeCache.txt").exists():
        parser.error("configure the project first; gemm.cu or build/CMakeCache.txt is missing")

    print(f"Testing {len(candidates)} configurations across {len(args.sizes)} shapes")
    original = SOURCE.read_text(encoding="utf-8")
    env = build_environment()
    records: list[dict[str, object]] = []

    try:
        for index, config in enumerate(candidates, 1):
            label = ", ".join(f"{key}={config[key]}" for key in ("BM", "BN", "BK", "TM", "TN"))
            print(f"[{index:02d}/{len(candidates):02d}] {label}", flush=True)
            SOURCE.write_text(apply_config(original, config), encoding="utf-8")
            built, build_output = build_profiler(env)
            if not built:
                records.append({"config": config, "status": "build_failed", "error": build_output[-4000:]})
                print("  build failed")
                continue

            timings: dict[str, float] = {}
            error = None
            for shape in args.sizes:
                shape_name = "x".join(map(str, shape))
                if not shape_is_compatible(shape, config):
                    continue
                try:
                    elapsed_us = benchmark(shape, args)
                    timings[shape_name] = elapsed_us
                    print(f"  {shape_name}: {elapsed_us:.3f} us")
                except RuntimeError as exc:
                    error = str(exc)
                    print(f"  {shape_name}: failed")
                    break

            records.append({
                "config": config,
                "status": "ok" if error is None else "run_failed",
                "timings_us": timings,
                **({"error": error} if error else {}),
            })
    finally:
        SOURCE.write_text(original, encoding="utf-8")
        build_profiler(env)

    winners: dict[str, dict[str, object]] = {}
    rankings: dict[str, list[dict[str, object]]] = {}
    for shape in args.sizes:
        name = "x".join(map(str, shape))
        ranked = [
            {"config": record["config"], "time_us": record["timings_us"][name]}
            for record in records
            if name in record.get("timings_us", {})
        ]
        ranked.sort(key=lambda item: item["time_us"])
        rankings[name] = ranked
        if ranked:
            winners[name] = ranked[0]

    result = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "constraint": "BM*BK == BN*BK == 4*threads; one float4 load per thread per input tile",
        "benchmark": {"warmup": args.warmup, "iterations": args.iterations, "repeats": args.repeats},
        "winners": winners,
        "rankings": rankings,
        "records": records,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\nWinners:")
    for shape, winner in winners.items():
        config = winner["config"]
        constants = ", ".join(f"{key}={config[key]}" for key in ("BM", "BN", "BK", "TM", "TN"))
        print(f"  {shape}: {winner['time_us']:.3f} us ({constants})")
    print(f"Results written to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
