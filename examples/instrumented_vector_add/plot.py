#!/usr/bin/env python3
"""Plot vector-add samples and label confirmed best completed configurations."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    if len(sys.argv) not in {2, 3}:
        print(f"usage: {Path(sys.argv[0]).name} SAMPLES.csv [PLOT.png]", file=sys.stderr)
        return 2

    samples_path = Path(sys.argv[1])
    plot_path = Path(sys.argv[2]) if len(sys.argv) == 3 else samples_path.with_name("runtime.png")
    with samples_path.open(newline="") as input_file:
        samples = list(csv.DictReader(input_file))
    if not samples:
        raise RuntimeError("The instrumentation CSV contains no samples.")

    times = [float(sample["time_seconds"]) for sample in samples]
    runtimes = [float(sample["runtime_microseconds"]) for sample in samples]
    estimates = [float(sample["candidate_estimate_microseconds"]) for sample in samples]
    # A median evolves while a candidate is being sampled. Comparing those
    # provisional estimates can make a higher estimate appear to be a new
    # best after the previous incumbent's estimate has moved upward. Keep the
    # final row for each candidate: it represents its completed measurement
    # budget and final robust estimate.
    final_by_candidate: dict[str, dict[str, str]] = {}
    for sample in samples:
        candidate = sample["candidate_index"]
        final_by_candidate[candidate] = sample

    confirmed_best_samples = []
    best_final_estimate = float("inf")
    for sample in sorted(final_by_candidate.values(), key=lambda item: float(item["time_seconds"])):
        final_estimate = float(sample["candidate_estimate_microseconds"])
        if final_estimate < best_final_estimate:
            best_final_estimate = final_estimate
            confirmed_best_samples.append(sample)

    figure, (axis, labels_axis) = plt.subplots(
        1,
        2,
        figsize=(20, 9),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [4.0, 1.25]},
    )
    axis.plot(times, runtimes, color="#9aa5b1", linewidth=0.5, alpha=0.38, label="raw context.tune runtime")
    axis.plot(
        times,
        estimates,
        color="#2a9d8f",
        linewidth=0.85,
        alpha=0.9,
        label="per-configuration MAD-filtered median",
    )
    axis.scatter(
        [float(sample["time_seconds"]) for sample in confirmed_best_samples],
        [float(sample["candidate_estimate_microseconds"]) for sample in confirmed_best_samples],
        color="#e45756",
        marker="*",
        s=90,
        zorder=3,
        label="confirmed best completed configuration",
    )
    for index, sample in enumerate(confirmed_best_samples, start=1):
        axis.annotate(
            str(index),
            (float(sample["time_seconds"]), float(sample["candidate_estimate_microseconds"])),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8},
        )
    axis.set_xlabel("elapsed wall time [s]")
    axis.set_ylabel("runtime [µs, logarithmic scale]")
    axis.set_yscale("log")
    axis.set_title("alpakaTune vector-add: raw launches and robust configuration history")
    axis.grid(alpha=0.25)
    axis.legend()
    labels_axis.axis("off")
    labels_axis.set_title("Confirmed robust bests", loc="left")
    label_lines = [
        (
            f"{index}. offset={sample['runtime_offset']}, SIMD={sample['simd_width']}, "
            f"blocks={sample['num_blocks']}\n"
            f"   t={float(sample['time_seconds']):.2f}s, "
            f"{float(sample['candidate_estimate_microseconds']):.1f} µs"
        )
        for index, sample in enumerate(confirmed_best_samples, start=1)
    ]
    labels_axis.text(
        0.0,
        1.0,
        "\n\n".join(label_lines),
        transform=labels_axis.transAxes,
        va="top",
        fontsize=8,
        family="monospace",
    )
    figure.savefig(plot_path, dpi=180)
    print(plot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
