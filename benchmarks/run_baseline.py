#!/usr/bin/env python3
"""Run the unmodified Alpaka examples as an instrumentation-free baseline."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import re
import statistics
import subprocess
import time


EXAMPLES = {
    "boundaryIter": "example/boundaryIter/boundaryIter",
    "grayScale": "example/grayScale/grayScale",
    "heatEquation2D": "example/heatEquation2D/heatEquation2D",
    "nBody": "example/nBody/nBody",
    "vectorAdd": "example/vectorAdd/vectorAdd",
}

RUNTIME_PATTERNS = {
    "grayScale": re.compile(r"Time for kernel execution \[s\]:\s*([0-9.eE+-]+)"),
    "vectorAdd": re.compile(r"Average time for kernel execution:\s*([0-9.eE+-]+)s"),
    "heatEquation2D": re.compile(r"Time per time step:\s*([0-9.eE+-]+)\s*ms"),
    "nBody": re.compile(r"Time per time step:\s*([0-9.eE+-]+)\s*ms"),
}

BASELINE_EXECUTORS = ("CpuOmpBlocks", "GpuCuda")


def _reported_executor(line: str) -> str | None:
    folded = line.casefold()
    if "cpuompblocks" in folded:
        return "CpuOmpBlocks"
    if "gpucuda" in folded or folded == "cuda":
        return "GpuCuda"
    if folded == "host":
        # HeatEquation2D and nBody print only the API name. The baseline build
        # contract disables CpuSerial, making Host unambiguously CpuOmpBlocks.
        return "CpuOmpBlocks"
    return None


def reported_runtimes(example: str, output: str) -> dict[str, float]:
    """Extract executor-specific upstream kernel/time-step averages."""
    pattern = RUNTIME_PATTERNS.get(example)
    if pattern is None:
        return {}
    current_executor = None
    runtimes: dict[str, float] = {}
    for line in output.splitlines():
        stripped = line.strip()
        folded = stripped.casefold()
        if folded.startswith("using native cpu"):
            current_executor = None
        elif stripped.startswith("Using alpaka accelerator:"):
            current_executor = _reported_executor(stripped)
        elif stripped.startswith("Running accelerator:"):
            current_executor = _reported_executor(
                stripped.split(":", maxsplit=1)[1].strip()
            )
        elif folded in {"cuda", "hip", "host", "oneapi"}:
            current_executor = _reported_executor(stripped)
        if current_executor is not None and (match := pattern.search(stripped)):
            runtime = float(match.group(1))
            if example in {"heatEquation2D", "nBody"}:
                runtime *= 1.0e-3
            # Retain the first Host value in old outputs that also contain
            # CpuSerial. New benchmark builds must compile CpuSerial out.
            runtimes.setdefault(current_executor, runtime)
    return runtimes


def reported_cuda_runtime(example: str, output: str) -> float | None:
    """Backward-compatible CUDA-only parser."""
    return reported_runtimes(example, output).get("GpuCuda")


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=repository / "build/alpaka-baseline",
        help="build directory configured from Alpaka's source tree",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--examples",
        nargs="+",
        choices=tuple(EXAMPLES),
        default=tuple(EXAMPLES),
    )
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.repetitions <= 0:
        parser.error("--repetitions must be greater than zero")
    if arguments.output is None:
        run_id = utc_now().strftime("%Y%m%dT%H%M%SZ")
        arguments.output = repository / "benchmarks/baseline-results" / run_id
    return arguments


def completed_run(path: Path) -> bool:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "completed"
    except (OSError, json.JSONDecodeError):
        return False


def run_once(executable: Path, directory: Path, example: str, repetition: int) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    command = [str(executable)]
    started = utc_now()
    started_monotonic = time.monotonic()
    try:
        with (directory / "stdout.log").open("w", encoding="utf-8") as stdout, \
             (directory / "stderr.log").open("w", encoding="utf-8") as stderr:
            result = subprocess.run(
                command,
                cwd=directory,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        return_code = result.returncode
        error = None
    except OSError as exception:
        return_code = None
        error = str(exception)
    duration = time.monotonic() - started_monotonic
    metadata = {
        "example": example,
        "repetition": repetition,
        "command": command,
        "started_at": started.isoformat(),
        "finished_at": utc_now().isoformat(),
        "duration_seconds": duration,
        "return_code": return_code,
        "status": "completed" if return_code == 0 else "failed",
    }
    if return_code == 0:
        try:
            runtimes = reported_runtimes(
                example, (directory / "stdout.log").read_text(encoding="utf-8")
            )
        except OSError:
            runtimes = {}
        metadata["reported_runtimes_seconds"] = runtimes
        metadata["reported_cuda_runtime_seconds"] = runtimes.get("GpuCuda")
    if error is not None:
        metadata["error"] = error
    write_json(directory / "run.json", metadata)
    return metadata


def main() -> int:
    arguments = parse_args()
    build_directory = arguments.build_dir.resolve()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "baseline.json",
        {
            "build_directory": str(build_directory),
            "examples": arguments.examples,
            "repetitions": arguments.repetitions,
            "created_at": utc_now().isoformat(),
            "note": (
                "These are the upstream Alpaka examples with no alpakaTune context. "
                "Enabled backends are determined by the baseline CMake configuration."
            ),
        },
    )

    failures: list[tuple[str, int]] = []
    summary: dict[str, dict] = {}
    for example in arguments.examples:
        executable = (build_directory / EXAMPLES[example]).resolve()
        durations: list[float] = []
        executor_runtimes: dict[str, list[float]] = {
            executor: [] for executor in BASELINE_EXECUTORS
        }
        for repetition in range(1, arguments.repetitions + 1):
            directory = output / example / f"{repetition:04d}"
            metadata_path = directory / "run.json"
            if arguments.resume and completed_run(metadata_path):
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                print(f"SKIP {example} / {repetition}", flush=True)
            else:
                if not executable.is_file():
                    print(f"Missing baseline executable: {executable}")
                    failures.append((example, repetition))
                    continue
                print(f"RUN  {example} / {repetition}", flush=True)
                metadata = run_once(executable, directory, example, repetition)
            if metadata["status"] == "completed":
                durations.append(metadata["duration_seconds"])
                reported = metadata.get("reported_runtimes_seconds")
                if not isinstance(reported, dict):
                    reported = {}
                    if metadata.get("reported_cuda_runtime_seconds") is not None:
                        reported["GpuCuda"] = metadata[
                            "reported_cuda_runtime_seconds"
                        ]
                for executor in BASELINE_EXECUTORS:
                    runtime = reported.get(executor)
                    if isinstance(runtime, (int, float)):
                        executor_runtimes[executor].append(runtime)
            else:
                failures.append((example, repetition))
        if durations:
            summary[example] = {
                "completed_repetitions": len(durations),
                "median_duration_seconds": statistics.median(durations),
                "minimum_duration_seconds": min(durations),
                "maximum_duration_seconds": max(durations),
                "durations_seconds": durations,
            }
            available_runtimes = {
                executor: runtimes
                for executor, runtimes in executor_runtimes.items()
                if runtimes
            }
            if available_runtimes:
                runtime_kind = (
                    "kernel"
                    if example in {"grayScale", "vectorAdd"}
                    else "time_step"
                )
                summary[example]["reported_runtime_kind"] = runtime_kind
                summary[example]["reported_runtimes"] = {}
                for executor, runtimes in available_runtimes.items():
                    steady_runtimes = runtimes[1:] if len(runtimes) > 1 else runtimes
                    summary[example]["reported_runtimes"][executor] = {
                        "warmup_runtime_seconds": runtimes[0],
                        "mean_runtime_seconds": statistics.fmean(steady_runtimes),
                        "runtimes_seconds": runtimes,
                    }

                # Keep the original CUDA summary fields readable by older
                # visualizers and imported-result tooling.
                cuda = summary[example]["reported_runtimes"].get("GpuCuda")
                if cuda is not None:
                    summary[example].update(
                        {
                            "warmup_reported_cuda_runtime_seconds": cuda[
                                "warmup_runtime_seconds"
                            ],
                            "mean_reported_cuda_runtime_seconds": cuda[
                                "mean_runtime_seconds"
                            ],
                            "reported_cuda_runtimes_seconds": cuda[
                                "runtimes_seconds"
                            ],
                        }
                    )
    write_json(output / "summary.json", summary)

    if failures:
        print("Failed baseline executions:")
        for example, repetition in failures:
            print(f"  {example} / {repetition}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
