#!/usr/bin/env python3
"""Run every tuned alpaka example with every tuning strategy."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
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
    "learned_hybrid",
)

LEARNED_STRATEGY = "learned_hybrid"
DEFAULT_STRATEGIES = tuple(
    strategy for strategy in STRATEGIES if strategy != LEARNED_STRATEGY
)
FNV1A_OFFSET_BASIS = 14_695_981_039_346_656_037
FNV1A_PRIME = 1_099_511_628_211
UINT64_MASK = (1 << 64) - 1

PROGRESS_INTERVAL_SECONDS = 30
DEFAULT_MAXIMUM_EXECUTIONS = 40_000
DEFAULT_MAXIMUM_RETIRED_CONFIGURATIONS = 100_000
# Full coverage is still application-bounded. This tuner guard prevents an
# incomplete exhaustive policy from extending the example indefinitely.
FULL_COVERAGE_MAXIMUM_EXECUTIONS = 1_000_000

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

TUNE_UNTIL_TERMINAL_EXAMPLE_ARGUMENTS = {
    "heatEquation2D": ("--tune-until-terminal",),
    "nBody": ("--tune-until-terminal",),
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


def benchmark_example_arguments(
    example: str, *, full_coverage: bool, tune_until_terminal: bool
) -> tuple[str, ...]:
    if full_coverage:
        return FULL_COVERAGE_EXAMPLE_ARGUMENTS.get(example, ())
    if tune_until_terminal:
        return TUNE_UNTIL_TERMINAL_EXAMPLE_ARGUMENTS.get(example, ())
    return ()


def model_digests(path: Path) -> tuple[str, str]:
    """Return provenance SHA-256 and the runtime's 64-bit FNV-1a digest."""
    sha256 = hashlib.sha256()
    fnv1a = FNV1A_OFFSET_BASIS
    with path.open("rb") as model:
        while chunk := model.read(64 * 1024):
            sha256.update(chunk)
            for value in chunk:
                fnv1a ^= value
                fnv1a = (fnv1a * FNV1A_PRIME) & UINT64_MASK
    return sha256.hexdigest(), f"{fnv1a:016x}"


def successful_run(
    directory: Path,
    require_full_coverage: bool = False,
    require_terminal_reason: bool = False,
    expected_model_sha256: str | None = None,
    expected_model_runtime_digest: str | None = None,
) -> bool:
    metadata = directory / "run.json"
    history = directory / "complete-history.json"
    if not metadata.exists() or not history.exists():
        return False
    try:
        value = json.loads(metadata.read_text(encoding="utf-8"))
        return (
            value.get("status") == "completed"
            and (not require_full_coverage or value.get("full_coverage_verified") is True)
            and (not require_terminal_reason or value.get("terminal_verified") is True)
            and (
                expected_model_runtime_digest is None
                or (
                    value.get("learning_verified") is True
                    and value.get("model_sha256") == expected_model_sha256
                    and value.get("model_runtime_digest")
                    == expected_model_runtime_digest
                )
            )
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
    model: Path | None = None,
    learned_candidate_pool_size: int | None = None,
    learned_candidate_batch_size: int | None = None,
) -> dict:
    configuration = copy.deepcopy(base_configuration)
    tuning = configuration["tuning"]
    # The benchmark runner deliberately uses the finite policy. Direct library
    # users inherit online_adaptive unless they make the same explicit choice.
    tuning["mode"] = "online_fixed"
    tuning["strategy"] = strategy
    tuning.pop("horizon", None)
    tuning.pop("horizon_offset_with_active_history", None)
    if full_coverage:
        tuning.update(FULL_COVERAGE_TUNING)
        tuning["maximum_executions"] = (
            FULL_COVERAGE_MAXIMUM_EXECUTIONS
            if maximum_executions is None
            else maximum_executions
        )
        tuning.pop("maximum_retired_configurations", None)
    else:
        tuning["maximum_executions"] = maximum_executions
        tuning["maximum_retired_configurations"] = maximum_retired_configurations
    learning = configuration.get("learning")
    if strategy == LEARNED_STRATEGY:
        if model is None:
            raise ValueError("learned_hybrid requires a model")
        if learning is None:
            learning = {}
            configuration["learning"] = learning
        if not isinstance(learning, dict):
            raise ValueError("configuration learning section must be a map")
        configuration["schema_version"] = 3
        learning["model"] = str(model.resolve())
        if learned_candidate_pool_size is not None:
            learning["candidate_pool_size"] = learned_candidate_pool_size
        if learned_candidate_batch_size is not None:
            learning["candidate_batch_size"] = learned_candidate_batch_size
    elif isinstance(learning, dict):
        learning.pop("model", None)
    configuration["schema_version"] = 3
    configuration.pop("persistence", None)
    configuration["history"] = {"read": False, "write": False}
    configuration["complete_history"] = {"file": str(history)}
    return configuration


def inspect_history(path: Path) -> dict:
    """Summarize persisted coverage without depending on a strategy implementation."""
    try:
        history = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exception:
        return {
            "valid": False,
            "all_contexts_complete": False,
            "all_contexts_terminal": False,
            "contexts": [],
            "messages": [f"cannot inspect history: {exception}"],
        }

    contexts = history.get("contexts")
    if not isinstance(contexts, dict) or not contexts:
        return {
            "valid": False,
            "all_contexts_complete": False,
            "all_contexts_terminal": False,
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
        "all_contexts_terminal": valid and bool(summaries) and all(
            summary["completion_reason"]
            in {
                "all_configurations",
                "maximum_executions",
                "maximum_retired_configurations",
            }
            for summary in summaries
        ),
        "contexts": summaries,
        "messages": messages,
    }


def inspect_learned_history(path: Path, expected_model_digest: str) -> dict:
    """Verify that every persisted context used the requested learned artifact."""
    try:
        history = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exception:
        return {
            "valid": False,
            "expected_model_digest": expected_model_digest,
            "contexts": [],
            "messages": [f"cannot inspect learned history: {exception}"],
        }

    contexts = history.get("contexts")
    if not isinstance(contexts, dict) or not contexts:
        return {
            "valid": False,
            "expected_model_digest": expected_model_digest,
            "contexts": [],
            "messages": ["history does not contain any learned tuning contexts"],
        }

    summaries: list[dict] = []
    messages: list[str] = []
    for fingerprint, context in contexts.items():
        metadata = context.get("metadata", {}) if isinstance(context, dict) else {}
        kernel = (
            metadata.get("kernel", fingerprint)
            if isinstance(metadata, dict)
            else fingerprint
        )
        learning = context.get("learning") if isinstance(context, dict) else None
        if not isinstance(learning, dict):
            messages.append(f"context {kernel} has no learning status")
            summaries.append(
                {
                    "fingerprint": fingerprint,
                    "kernel": kernel,
                    "valid": False,
                }
            )
            continue

        status = learning.get("status")
        artifact_load_status = learning.get("artifact_load_status")
        model_digest = learning.get("model_digest")
        valid = (
            status == "active"
            and artifact_load_status == "available"
            and model_digest == expected_model_digest
        )
        if status != "active":
            messages.append(
                f"context {kernel} learned strategy status is {status!r}, expected 'active'"
            )
        if artifact_load_status != "available":
            messages.append(
                f"context {kernel} artifact load status is "
                f"{artifact_load_status!r}, expected 'available'"
            )
        if model_digest != expected_model_digest:
            messages.append(
                f"context {kernel} model digest is {model_digest!r}, expected "
                f"{expected_model_digest!r}"
            )
        summaries.append(
            {
                "fingerprint": fingerprint,
                "kernel": kernel,
                "status": status,
                "artifact_load_status": artifact_load_status,
                "model_file": learning.get("model_file"),
                "model_digest": model_digest,
                "valid": valid,
            }
        )

    return {
        "valid": bool(summaries) and all(summary["valid"] for summary in summaries),
        "expected_model_digest": expected_model_digest,
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
    parser.add_argument(
        "--model",
        type=Path,
        help="trained .atml artifact; required when learned_hybrid is selected",
    )
    parser.add_argument(
        "--learned-candidate-pool-size",
        type=int,
        help="bounded learned candidate pool capacity (default: tuner configuration)",
    )
    parser.add_argument(
        "--learned-candidate-batch-size",
        type=int,
        help="learned scoring batch size (default: tuner configuration)",
    )
    parser.add_argument(
        "--backend",
        help="limit examples to api:deviceKind, e.g. cuda:nvidiaGpu or host:cpu",
    )
    parser.add_argument(
        "--executor",
        help="limit examples to one executor, e.g. gpuCuda or cpuOmpBlocks",
    )
    parser.add_argument("--maximum-executions", type=int)
    parser.add_argument("--maximum-retired-configurations", type=int)
    parser.add_argument(
        "--full-coverage",
        action="store_true",
        help=(
            "collect the complete exhaustive surface with three measured runs per legal "
            "candidate within a configurable tuner execution guard"
        ),
    )
    parser.add_argument(
        "--tune-until-terminal",
        action="store_true",
        help=(
            "keep finite simulations running through the example minimum and "
            "each tuner's policy goal (legacy option name)"
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
        if arguments.tune_until_terminal:
            parser.error("--tune-until-terminal cannot be combined with --full-coverage")
        if arguments.strategies is not None and arguments.strategies != ["exhaustive"]:
            parser.error("--full-coverage only supports --strategies exhaustive")
        if arguments.maximum_retired_configurations is not None:
            parser.error(
                "--full-coverage cannot be combined with "
                "--maximum-retired-configurations"
            )
        if arguments.maximum_executions is None:
            arguments.maximum_executions = FULL_COVERAGE_MAXIMUM_EXECUTIONS
        arguments.strategies = ("exhaustive",)
    else:
        arguments.strategies = tuple(arguments.strategies or DEFAULT_STRATEGIES)
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
    if LEARNED_STRATEGY in arguments.strategies and arguments.model is None:
        parser.error("--model is required when learned_hybrid is selected")
    learned_sizes = (
        arguments.learned_candidate_pool_size,
        arguments.learned_candidate_batch_size,
    )
    if any(value is not None and value <= 0 for value in learned_sizes):
        parser.error(
            "learned candidate pool and batch sizes must be greater than zero"
        )
    if (
        any(value is not None for value in learned_sizes)
        and LEARNED_STRATEGY not in arguments.strategies
    ):
        parser.error("learned candidate pool and batch sizes require learned_hybrid")
    if (
        all(value is not None for value in learned_sizes)
        and arguments.learned_candidate_batch_size
        > arguments.learned_candidate_pool_size
    ):
        parser.error("learned candidate batch size must not exceed pool size")
    if arguments.model is not None:
        arguments.model = arguments.model.expanduser().resolve()
        if not arguments.model.is_file():
            parser.error(f"--model is not a readable file: {arguments.model}")
        try:
            with arguments.model.open("rb") as model:
                model.read(1)
        except OSError as exception:
            parser.error(f"cannot read --model {arguments.model}: {exception}")
        if LEARNED_STRATEGY not in arguments.strategies:
            parser.error("--model requires --strategies learned_hybrid")
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
    require_terminal_reason: bool,
    model: Path | None = None,
    model_sha256: str | None = None,
    model_runtime_digest: str | None = None,
    learned_candidate_pool_size: int | None = None,
    learned_candidate_batch_size: int | None = None,
) -> bool:
    directory = output / example / strategy
    directory.mkdir(parents=True, exist_ok=True)
    history = (directory / "complete-history.json").resolve()
    configuration_path = (directory / "tuning.yaml").resolve()

    configuration = benchmark_configuration(
        base_configuration,
        strategy,
        history,
        maximum_executions,
        maximum_retired_configurations,
        full_coverage,
        model,
        learned_candidate_pool_size,
        learned_candidate_batch_size,
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
    if strategy == LEARNED_STRATEGY:
        metadata.update(
            {
                "model": str(model),
                "model_sha256": model_sha256,
                "model_runtime_digest": model_runtime_digest,
            }
        )
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
    learning_diagnostics = (
        inspect_learned_history(history, model_runtime_digest)
        if history_present
        and strategy == LEARNED_STRATEGY
        and model_runtime_digest is not None
        else None
    )
    learning_verified = bool(
        learning_diagnostics is not None and learning_diagnostics["valid"]
    )
    full_coverage_verified = bool(
        diagnostics is not None and diagnostics["all_contexts_complete"]
    )
    terminal_verified = bool(
        diagnostics is not None and diagnostics["all_contexts_terminal"]
    )
    status = (
        "completed"
        if return_code == 0
        and history_present
        and (not full_coverage or full_coverage_verified)
        and (not require_terminal_reason or terminal_verified)
        and (strategy != LEARNED_STRATEGY or learning_verified)
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
            "terminal_verified": (
                terminal_verified if require_terminal_reason else None
            ),
            "learning_validation": learning_diagnostics,
            "learning_verified": (
                learning_verified if strategy == LEARNED_STRATEGY else None
            ),
            "status": status,
        }
    )
    if error is not None:
        metadata["error"] = error
    write_json(directory / "run.json", metadata)
    if strategy == "exhaustive" and diagnostics is not None:
        for message in diagnostics["messages"]:
            print(f"WARN {example} / {strategy}: {message}", file=sys.stderr, flush=True)
    if strategy == LEARNED_STRATEGY and learning_diagnostics is not None:
        for message in learning_diagnostics["messages"]:
            print(f"ERROR {example} / {strategy}: {message}", file=sys.stderr, flush=True)
    return status == "completed"


def main() -> int:
    arguments = parse_args()
    repository = Path(__file__).resolve().parents[1]
    build_directory = arguments.build_dir.resolve()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    model_sha256 = None
    model_runtime_digest = None
    if arguments.model is not None:
        try:
            model_sha256, model_runtime_digest = model_digests(arguments.model)
        except OSError as exception:
            print(f"Cannot read learned model: {exception}", file=sys.stderr)
            return 2

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
            "model": str(arguments.model) if arguments.model is not None else None,
            "model_sha256": model_sha256,
            "model_runtime_digest": model_runtime_digest,
            "learned_candidate_pool_size": arguments.learned_candidate_pool_size,
            "learned_candidate_batch_size": arguments.learned_candidate_batch_size,
            "maximum_executions": arguments.maximum_executions,
            "maximum_retired_configurations": arguments.maximum_retired_configurations,
            "full_coverage": arguments.full_coverage,
            "tune_until_terminal": arguments.tune_until_terminal,
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
        command.extend(
            benchmark_example_arguments(
                example,
                full_coverage=arguments.full_coverage,
                tune_until_terminal=arguments.tune_until_terminal,
            )
        )
        for strategy in arguments.strategies:
            directory = output / example / strategy
            if arguments.resume and successful_run(
                directory,
                require_full_coverage=arguments.full_coverage,
                require_terminal_reason=arguments.tune_until_terminal,
                expected_model_sha256=(
                    model_sha256 if strategy == LEARNED_STRATEGY else None
                ),
                expected_model_runtime_digest=(
                    model_runtime_digest if strategy == LEARNED_STRATEGY else None
                ),
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
                arguments.tune_until_terminal,
                arguments.model if strategy == LEARNED_STRATEGY else None,
                model_sha256 if strategy == LEARNED_STRATEGY else None,
                model_runtime_digest if strategy == LEARNED_STRATEGY else None,
                (
                    arguments.learned_candidate_pool_size
                    if strategy == LEARNED_STRATEGY
                    else None
                ),
                (
                    arguments.learned_candidate_batch_size
                    if strategy == LEARNED_STRATEGY
                    else None
                ),
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
