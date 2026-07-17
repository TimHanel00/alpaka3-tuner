#!/usr/bin/env python3
"""Run every tuned alpaka example with every tuning strategy."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml


STRATEGIES = (
    "exhaustive",
    "random",
    "simulated_annealing",
    "bayesian_optimization",
)

PROGRESS_INTERVAL_SECONDS = 30
DEFAULT_MAXIMUM_EXECUTIONS = 100_000
DEFAULT_MAXIMUM_RETIRED_CONFIGURATIONS = 100_000

FULL_COVERAGE_TUNING = {
    "warmup_runs": 1,
    "runs_per_candidate": 3,
    "minimum_runs_per_candidate": 3,
    "mann_whitney_early_stop": False,
    "max_consecutive_runs": 4,
}

FULL_COVERAGE_EXAMPLE_ARGUMENTS = {
    "heatEquation2D": ("--tune-until-complete",),
    "nBody": ("--tune-until-complete",),
}

EXAMPLES = {
    "boundaryIter": ("example/boundaryIter/alpakaTune_boundaryIter",),
    "grayScale": ("example/grayScale/alpakaTune_grayScale",),
    "heatEquation2D": ("example/heatEquation2D/alpakaTune_heatEquation2D",),
    "matrixMultiplication": (
        "example/matrixMultiplication/alpakaTune_matrixMultiplication",
    ),
    "nBody": ("example/nBody/alpakaTune_nBody",),
    "randomInit": ("example/randomInit/alpakaTune_randomInit",),
    "scan": ("example/scan/alpakaTune_scan",),
    "scan_improved": ("example/scan/alpakaTune_scan_improved",),
    "vectorAdd": ("example/vectorAdd/alpakaTune_vectorAdd",),
    "tutorial_05_kernel": ("example/tutorial/alpakaTune_05_kernel",),
    "tutorial_06_cudaLikeKernel": ("example/tutorial/alpakaTune_06_cudaLikeKernel",),
    "tutorial_07_chunkedData": ("example/tutorial/alpakaTune_07_chunkedData",),
}

EXAMPLE_ALIASES = {
    "greyScale": "grayScale",
}


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def successful_run(directory: Path, require_full_coverage: bool = False) -> bool:
    metadata = directory / "run.json"
    history = directory / "history.json"
    if not metadata.exists() or not history.exists():
        return False
    try:
        value = json.loads(metadata.read_text(encoding="utf-8"))
        return value.get("status") == "completed" and (
            not require_full_coverage or value.get("full_coverage_verified") is True
        )
    except (OSError, json.JSONDecodeError):
        return False


def normalize_examples(
    parser: argparse.ArgumentParser,
    values: list[str] | tuple[str, ...],
    option: str,
) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        example = EXAMPLE_ALIASES.get(value, value)
        if example not in EXAMPLES:
            parser.error(
                f"unknown example for {option}: {value} "
                f"(choose from {', '.join(EXAMPLES)})"
            )
        if example not in normalized:
            normalized.append(example)
    return tuple(normalized)


def benchmark_configuration(
    base_configuration: dict,
    strategy: str,
    history: Path,
    maximum_executions: int | None,
    maximum_retired_configurations: int | None,
    full_coverage: bool,
) -> dict:
    configuration = copy.deepcopy(base_configuration)
    tuning = configuration["tuning"]
    tuning["strategy"] = strategy
    if full_coverage:
        tuning.update(FULL_COVERAGE_TUNING)
        tuning.pop("maximum_executions", None)
        tuning.pop("maximum_retired_configurations", None)
    else:
        tuning["maximum_executions"] = maximum_executions
        tuning["maximum_retired_configurations"] = maximum_retired_configurations
    configuration["persistence"] = {"file": str(history)}
    return configuration


def inspect_history(path: Path) -> dict:
    """Summarize persisted coverage without depending on a strategy implementation."""
    try:
        history = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exception:
        return {
            "valid": False,
            "all_contexts_complete": False,
            "contexts": [],
            "messages": [f"cannot inspect history: {exception}"],
        }

    contexts = history.get("contexts")
    if not isinstance(contexts, dict) or not contexts:
        return {
            "valid": False,
            "all_contexts_complete": False,
            "contexts": [],
            "messages": ["history does not contain any tuning contexts"],
        }

    summaries: list[dict] = []
    messages: list[str] = []
    valid = True
    for fingerprint, context in contexts.items():
        if not isinstance(context, dict):
            valid = False
            messages.append(f"context {fingerprint} is not a JSON object")
            continue
        candidate_count = context.get("candidate_count")
        retired_count = context.get("retired_configuration_count")
        rejected = context.get("rejected_candidates")
        reason = context.get("completion_reason", "none")
        metadata = context.get("metadata", {})
        kernel = metadata.get("kernel", fingerprint) if isinstance(metadata, dict) else fingerprint

        counts_valid = (
            isinstance(candidate_count, int)
            and candidate_count >= 0
            and isinstance(retired_count, int)
            and retired_count >= 0
            and isinstance(rejected, list)
            and len(rejected) == candidate_count
            and all(isinstance(value, bool) for value in rejected)
        )
        rejected_count = sum(rejected) if counts_valid else None
        legal_count = candidate_count - rejected_count if counts_valid else None
        accounted_count = retired_count + rejected_count if counts_valid else None
        coverage = (
            retired_count / legal_count
            if counts_valid and legal_count not in (None, 0)
            else (1.0 if counts_valid and legal_count == 0 else None)
        )
        complete = (
            counts_valid
            and reason == "all_configurations"
            and accounted_count == candidate_count
        )
        if not counts_valid:
            valid = False
            messages.append(f"context {kernel} has invalid coverage counters")
        elif reason == "maximum_executions":
            messages.append(
                f"context {kernel} was capped by maximum_executions after "
                f"{retired_count}/{legal_count} legal candidates ({coverage:.1%}); "
                "the execution budget counts warm-ups and repeated measurements, "
                "so this does not indicate skipped exhaustive candidates"
            )
        elif reason == "maximum_retired_configurations":
            messages.append(
                f"context {kernel} was capped by maximum_retired_configurations after "
                f"{retired_count}/{legal_count} legal candidates ({coverage:.1%})"
            )
        elif reason != "all_configurations":
            messages.append(
                f"context {kernel} exited before tuning completed after "
                f"{retired_count}/{legal_count} legal candidates ({coverage:.1%})"
            )
        elif accounted_count != candidate_count:
            messages.append(
                f"context {kernel} reports all_configurations but accounts for "
                f"{accounted_count}/{candidate_count} candidates"
            )

        summaries.append(
            {
                "fingerprint": fingerprint,
                "kernel": kernel,
                "candidate_count": candidate_count,
                "legal_candidate_count": legal_count,
                "rejected_candidate_count": rejected_count,
                "retired_configuration_count": retired_count,
                "coverage": coverage,
                "completion_reason": reason,
                "complete": complete,
            }
        )

    return {
        "valid": valid,
        "all_contexts_complete": valid and bool(summaries) and all(
            summary["complete"] for summary in summaries
        ),
        "contexts": summaries,
        "messages": messages,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=repository / "build")
    parser.add_argument("--config", type=Path, default=repository / "config/alpakaTune.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--examples",
        nargs="+",
        default=tuple(EXAMPLES),
        help="run only these benchmark examples (greyScale is accepted as an alias)",
    )
    parser.add_argument(
        "--exclude-examples",
        nargs="+",
        default=(),
        help="exclude these benchmark examples after applying --examples",
    )
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES)
    parser.add_argument("--backend", help="limit examples to api:deviceKind, e.g. cuda:nvidiaGpu")
    parser.add_argument("--executor", help="limit examples to one executor, e.g. gpuCuda")
    parser.add_argument("--maximum-executions", type=int)
    parser.add_argument("--maximum-retired-configurations", type=int)
    parser.add_argument(
        "--full-coverage",
        action="store_true",
        help=(
            "collect the complete exhaustive surface with three measured runs per legal "
            "candidate and no tuner-wide completion limits"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    arguments = parser.parse_args(argv)
    selected_examples = normalize_examples(parser, arguments.examples, "--examples")
    arguments.exclude_examples = normalize_examples(
        parser, arguments.exclude_examples, "--exclude-examples"
    )
    excluded_examples = set(arguments.exclude_examples)
    arguments.examples = tuple(
        example for example in selected_examples if example not in excluded_examples
    )
    if not arguments.examples:
        parser.error("the example selection is empty")
    if arguments.full_coverage:
        if arguments.strategies is not None and arguments.strategies != ["exhaustive"]:
            parser.error("--full-coverage only supports --strategies exhaustive")
        if (
            arguments.maximum_executions is not None
            or arguments.maximum_retired_configurations is not None
        ):
            parser.error("--full-coverage cannot be combined with completion limits")
        arguments.strategies = ("exhaustive",)
    else:
        arguments.strategies = tuple(arguments.strategies or STRATEGIES)
        if arguments.maximum_executions is None:
            arguments.maximum_executions = DEFAULT_MAXIMUM_EXECUTIONS
        if arguments.maximum_retired_configurations is None:
            arguments.maximum_retired_configurations = (
                DEFAULT_MAXIMUM_RETIRED_CONFIGURATIONS
            )
    if (
        arguments.maximum_executions is not None
        and arguments.maximum_executions <= 0
    ) or (
        arguments.maximum_retired_configurations is not None
        and arguments.maximum_retired_configurations <= 0
    ):
        parser.error("completion limits must be greater than zero")
    if arguments.output is None:
        run_id = utc_now().strftime("%Y%m%dT%H%M%SZ")
        arguments.output = repository / "benchmarks/results" / run_id
    return arguments


def run_pair(
    example: str,
    strategy: str,
    command: list[str],
    base_configuration: dict,
    output: Path,
    maximum_executions: int | None,
    maximum_retired_configurations: int | None,
    full_coverage: bool,
) -> bool:
    directory = output / example / strategy
    directory.mkdir(parents=True, exist_ok=True)
    history = (directory / "history.json").resolve()
    configuration_path = (directory / "tuning.yaml").resolve()

    configuration = benchmark_configuration(
        base_configuration,
        strategy,
        history,
        maximum_executions,
        maximum_retired_configurations,
        full_coverage,
    )
    history.unlink(missing_ok=True)
    configuration_path.write_text(yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8")

    started = utc_now()
    started_monotonic = time.monotonic()
    metadata = {
        "example": example,
        "strategy": strategy,
        "command": command,
        "configuration": str(configuration_path),
        "history": str(history),
        "started_at": started.isoformat(),
        "started_at_unix_seconds": started.timestamp(),
        "status": "running",
    }
    write_json(directory / "run.json", metadata)

    environment = os.environ.copy()
    environment["ALPAKA_TUNE_CONFIG"] = str(configuration_path)
    try:
        with (directory / "stdout.log").open("w", encoding="utf-8") as stdout, \
             (directory / "stderr.log").open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(
                command,
                cwd=directory,
                env=environment,
                stdout=stdout,
                stderr=stderr,
            )
            while True:
                try:
                    return_code = process.wait(timeout=PROGRESS_INTERVAL_SECONDS)
                    break
                except subprocess.TimeoutExpired:
                    elapsed_seconds = time.monotonic() - started_monotonic
                    history_state = "history written" if history.exists() else "no history yet"
                    print(
                        f"WAIT {example} / {strategy} "
                        f"({elapsed_seconds:.0f}s elapsed, pid {process.pid}, {history_state})",
                        flush=True,
                    )
        error = None
    except OSError as exception:
        return_code = None
        error = str(exception)

    finished = utc_now()
    history_present = history.exists()
    diagnostics = inspect_history(history) if history_present else None
    full_coverage_verified = bool(
        diagnostics is not None and diagnostics["all_contexts_complete"]
    )
    status = (
        "completed"
        if return_code == 0
        and history_present
        and (not full_coverage or full_coverage_verified)
        else "failed"
    )
    metadata.update(
        {
            "finished_at": finished.isoformat(),
            "finished_at_unix_seconds": finished.timestamp(),
            "duration_seconds": time.monotonic() - started_monotonic,
            "return_code": return_code,
            "history_present": history_present,
            "history_diagnostics": diagnostics,
            "full_coverage_verified": full_coverage_verified if full_coverage else None,
            "status": status,
        }
    )
    if error is not None:
        metadata["error"] = error
    write_json(directory / "run.json", metadata)
    if strategy == "exhaustive" and diagnostics is not None:
        for message in diagnostics["messages"]:
            print(f"WARN {example} / {strategy}: {message}", file=sys.stderr, flush=True)
    return status == "completed"


def main() -> int:
    arguments = parse_args()
    repository = Path(__file__).resolve().parents[1]
    build_directory = arguments.build_dir.resolve()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    try:
        configuration = yaml.safe_load(arguments.config.resolve().read_text(encoding="utf-8"))
        if not isinstance(configuration, dict) or not isinstance(configuration.get("tuning"), dict):
            raise ValueError("configuration has no tuning map")
    except (OSError, ValueError, yaml.YAMLError) as exception:
        print(f"Cannot load benchmark configuration: {exception}", file=sys.stderr)
        return 2

    warmup_runs = configuration["tuning"].get("warmup_runs", 0)
    if not isinstance(warmup_runs, int) or warmup_runs < 0:
        print("Cannot load benchmark configuration: tuning.warmup_runs must be a non-negative integer", file=sys.stderr)
        return 2
    if (
        arguments.maximum_executions is not None
        and arguments.maximum_executions <= warmup_runs
    ):
        print(
            "--maximum-executions must exceed tuning.warmup_runs "
            f"({warmup_runs}) so that at least one timing sample is recorded.",
            file=sys.stderr,
        )
        return 2

    write_json(
        output / "benchmark.json",
        {
            "repository": str(repository),
            "build_directory": str(build_directory),
            "source_configuration": str(arguments.config.resolve()),
            "examples": arguments.examples,
            "excluded_examples": arguments.exclude_examples,
            "strategies": arguments.strategies,
            "maximum_executions": arguments.maximum_executions,
            "maximum_retired_configurations": arguments.maximum_retired_configurations,
            "full_coverage": arguments.full_coverage,
            "measurement_policy": FULL_COVERAGE_TUNING if arguments.full_coverage else None,
            "backend": arguments.backend,
            "executor": arguments.executor,
            "created_at": utc_now().isoformat(),
        },
    )

    failures: list[tuple[str, str]] = []
    for example in arguments.examples:
        executable, *extra_arguments = EXAMPLES[example]
        command = [str((build_directory / executable).resolve()), *extra_arguments]
        if arguments.backend is not None:
            command.extend(["--backend", arguments.backend])
        if arguments.executor is not None:
            command.extend(["--executor", arguments.executor])
        if arguments.full_coverage:
            command.extend(FULL_COVERAGE_EXAMPLE_ARGUMENTS.get(example, ()))
        for strategy in arguments.strategies:
            directory = output / example / strategy
            if arguments.resume and successful_run(
                directory, require_full_coverage=arguments.full_coverage
            ):
                print(f"SKIP {example} / {strategy}", flush=True)
                continue
            print(f"RUN  {example} / {strategy}", flush=True)
            if not run_pair(
                example,
                strategy,
                command,
                configuration,
                output,
                arguments.maximum_executions,
                arguments.maximum_retired_configurations,
                arguments.full_coverage,
            ):
                failures.append((example, strategy))
                print(f"FAIL {example} / {strategy}", file=sys.stderr, flush=True)

    if not arguments.no_plot:
        plotter = Path(__file__).with_name("visualize.py")
        plot = subprocess.run([sys.executable, str(plotter), str(output)], check=False)
        if plot.returncode != 0:
            failures.append(("visualizer", "all"))

    if failures:
        print("Failed benchmark pairs:", file=sys.stderr)
        for example, strategy in failures:
            print(f"  {example} / {strategy}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
