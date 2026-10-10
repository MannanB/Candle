#!/usr/bin/env python3
"""Create an Nsight Compute diagnostic dashboard for every GEMM version."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import io
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

from autotune_gemm import PROFILER, ROOT, build_environment, build_profiler
from plot_gemm_benchmarks import KERNELS, TIME_RE


SECTIONS = (
    "SpeedOfLight", "ComputeWorkloadAnalysis", "MemoryWorkloadAnalysis",
    "Occupancy", "SchedulerStats", "WarpStateStats", "InstructionStats",
    "LaunchStats",
)


def ncu_executable() -> Path:
    found = shutil.which("ncu")
    if not found:
        raise RuntimeError("Nsight Compute CLI (ncu) was not found on PATH")
    path = Path(found)
    executable = path.with_suffix(".exe")
    return executable if executable.exists() else path


def clean_benchmark(kernel: str, size: int, args: argparse.Namespace) -> float:
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
            raise RuntimeError(output)
        match = TIME_RE.search(output)
        if not match:
            raise RuntimeError(f"could not parse profiler output:\n{output}")
        samples.append(float(match.group(1)))
    return statistics.median(samples)


def numeric(value: str) -> float:
    return float(value.replace(",", ""))


def collect_ncu(kernel: str, size: int, report_base: Path) -> dict[str, object]:
    command = [str(ncu_executable()), "--force-overwrite", "-o", str(report_base)]
    for section in SECTIONS:
        command.extend(("--section", section))
    command.extend((
        str(PROFILER), kernel,
        "--rows", str(size), "--inner", str(size), "--cols", str(size),
    ))
    run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if run.returncode:
        raise RuntimeError(run.stdout + run.stderr)

    imported = subprocess.run(
        [str(ncu_executable()), "--import", str(report_base) + ".ncu-rep", "--csv", "--page", "details"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if imported.returncode:
        raise RuntimeError(imported.stdout + imported.stderr)

    metrics: dict[tuple[str, str], float] = {}
    rule_descriptions = []
    rows = list(csv.DictReader(io.StringIO(imported.stdout)))
    for row in rows:
        section = row.get("Section Name", "")
        name = row.get("Metric Name", "")
        value = row.get("Metric Value", "")
        if section and name and value:
            try:
                metrics[(section, name)] = numeric(value)
            except ValueError:
                pass
        description = row.get("Rule Description", "")
        if description:
            rule_descriptions.append(description)

    dominant_stall_cycles = 0.0
    dominant_stall_reason = "not reported"
    stall_pattern = re.compile(
        r"spends\s+([0-9.]+)\s+cycles being stalled waiting for the\s+(.+?)\s+(?:instruction queue|memory)",
        re.IGNORECASE,
    )
    for description in rule_descriptions:
        match = stall_pattern.search(description)
        if match and float(match.group(1)) > dominant_stall_cycles:
            dominant_stall_cycles = float(match.group(1))
            dominant_stall_reason = match.group(2).strip()

    def metric(section: str, name: str) -> float:
        return metrics.get((section, name), 0.0)

    return {
        "achieved_occupancy_pct": metric("Occupancy", "Achieved Occupancy"),
        "theoretical_occupancy_pct": metric("Occupancy", "Theoretical Occupancy"),
        "registers_per_thread": metric("Launch Statistics", "Registers Per Thread"),
        "static_shared_kb": metric("Launch Statistics", "Static Shared Memory Per Block"),
        "executed_instructions": metric("Instruction Statistics", "Executed Instructions"),
        "issue_slots_busy_pct": metric("Compute Workload Analysis", "Issue Slots Busy"),
        "no_eligible_pct": metric("Scheduler Statistics", "No Eligible"),
        "compute_throughput_pct": metric("GPU Speed Of Light Throughput", "Compute (SM) Throughput"),
        "memory_throughput_pct": metric("GPU Speed Of Light Throughput", "Memory Throughput"),
        "warp_cycles_per_instruction": metric("Warp State Statistics", "Warp Cycles Per Issued Instruction"),
        "l1_hit_pct": metric("Memory Workload Analysis", "L1/TEX Hit Rate"),
        "l2_hit_pct": metric("Memory Workload Analysis", "L2 Hit Rate"),
        "waves_per_sm": metric("Launch Statistics", "Waves Per SM"),
        "dominant_stall_cycles": dominant_stall_cycles,
        "dominant_stall_reason": dominant_stall_reason,
        "ncu_report": str(report_base) + ".ncu-rep",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark_results" / "diagnostics")
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()

    if args.size <= 0 or args.size % 128:
        parser.error("--size must be a positive multiple of 128 for every current kernel")
    if not args.no_build:
        ok, output = build_profiler(build_environment())
        if not ok:
            raise RuntimeError(f"profiler build failed:\n{output}")

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy and matplotlib are required; run scripts\\setup.bat") from exc

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    reports_dir = output_dir / "ncu_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for label, kernel in KERNELS.items():
        print(f"Profiling {label}...", flush=True)
        elapsed_us = clean_benchmark(kernel, args.size, args)
        counters = collect_ncu(kernel, args.size, reports_dir / f"{kernel}_{args.size}")
        results.append({
            "label": label,
            "kernel": kernel,
            "size": args.size,
            "time_us": elapsed_us,
            "tflops": (2.0 * args.size**3) / (elapsed_us * 1.0e6),
            **counters,
        })

    json_path = output_dir / f"gemm_diagnostics_{args.size}.json"
    csv_path = output_dir / f"gemm_diagnostics_{args.size}.csv"
    png_path = output_dir / f"gemm_diagnostics_{args.size}.png"
    json_path.write_text(json.dumps({
        "generated_at": datetime.now().astimezone().isoformat(),
        "shape": [args.size, args.size, args.size],
        "results": results,
    }, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    labels = [result["label"] for result in results]
    colors = plt.get_cmap("tab10")(np.arange(len(labels)))
    x = np.arange(len(labels))
    figure, axes = plt.subplots(4, 3, figsize=(18, 18))

    def bars(axis, key, title, ylabel, scale=1.0):
        values = [float(result[key]) / scale for result in results]
        containers = axis.bar(x, values, color=colors)
        axis.bar_label(containers, fmt="%.2f", fontsize=8, padding=2)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)

    bars(axes[0, 0], "tflops", "Clean throughput", "TFLOP/s")
    bars(axes[0, 1], "time_us", "Clean kernel time", "µs")
    width = 0.38
    axes[0, 2].bar(x - width / 2, [r["achieved_occupancy_pct"] for r in results], width, label="Achieved")
    axes[0, 2].bar(x + width / 2, [r["theoretical_occupancy_pct"] for r in results], width, label="Theoretical")
    axes[0, 2].set_title("Occupancy"); axes[0, 2].set_ylabel("%"); axes[0, 2].legend()
    bars(axes[1, 0], "registers_per_thread", "Register pressure", "registers/thread")
    bars(axes[1, 1], "static_shared_kb", "Static shared memory", "KiB/block")
    bars(axes[1, 2], "executed_instructions", "Executed instructions", "millions", 1.0e6)
    axes[2, 0].bar(x - width / 2, [r["issue_slots_busy_pct"] for r in results], width, label="Issue slots busy")
    axes[2, 0].bar(x + width / 2, [r["no_eligible_pct"] for r in results], width, label="No eligible warp")
    axes[2, 0].set_title("Scheduler utilization"); axes[2, 0].set_ylabel("%"); axes[2, 0].legend()
    axes[2, 1].bar(x - width / 2, [r["compute_throughput_pct"] for r in results], width, label="Compute")
    axes[2, 1].bar(x + width / 2, [r["memory_throughput_pct"] for r in results], width, label="Memory")
    axes[2, 1].set_title("Speed-of-light utilization"); axes[2, 1].set_ylabel("%"); axes[2, 1].legend()
    bars(axes[2, 2], "warp_cycles_per_instruction", "Warp issue latency", "cycles/issued instruction")
    axes[3, 0].bar(x - width / 2, [r["l1_hit_pct"] for r in results], width, label="L1/TEX")
    axes[3, 0].bar(x + width / 2, [r["l2_hit_pct"] for r in results], width, label="L2")
    axes[3, 0].set_title("Cache hit rates"); axes[3, 0].set_ylabel("%"); axes[3, 0].legend()
    bars(axes[3, 1], "waves_per_sm", "Grid depth", "waves/SM")
    bars(axes[3, 2], "dominant_stall_cycles", "Dominant reported stall", "cycles/instruction")
    for index, result in enumerate(results):
        axes[3, 2].text(index, result["dominant_stall_cycles"], result["dominant_stall_reason"],
                        rotation=90, ha="center", va="bottom", fontsize=7)

    for axis in axes.flat:
        axis.set_xticks(x, labels, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(f"FP32 GEMM Diagnostic Dashboard — {args.size}³", fontsize=18)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(png_path, dpi=180)
    plt.close(figure)

    print(f"Dashboard: {png_path}")
    print(f"CSV:       {csv_path}")
    print(f"JSON:      {json_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
