#!/usr/bin/env python3
"""Plot measured candidate runtimes from a benchmark result tree."""

from __future__ import annotations

import argparse
from collections import defaultdict
import html
import json
import math
from pathlib import Path
import re
import statistics
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STRATEGY_LABELS = {
    "exhaustive": "Exhaustive",
    "random": "Random",
    "simulated_annealing": "Simulated annealing",
    "bayesian_optimization": "Bayesian optimization",
    "learned_hybrid": "Learned hybrid",
}

BASELINE_STYLES = {
    "CpuOmpBlocks": {"color": "#555555", "linestyle": ":"},
    "GpuCuda": {"color": "black", "linestyle": "--"},
}


def context_key(cache: dict) -> tuple:
    metadata = cache.get("metadata", {})
    return (
        tuple(metadata.get("identity_entries", ())),
        metadata.get("kernel", "unknown kernel"),
        metadata.get("device", "unknown device"),
        metadata.get("launch_specification", "unknown launch"),
        cache.get("candidate_count", 0),
    )


def context_name(key: tuple) -> str:
    identities, kernel, device, launch, candidate_count = key
    names = [entry.split("=", 1)[-1] for entry in identities]
    return ", ".join(names) if names else str(kernel)


def context_details(key: tuple) -> str:
    _identities, _kernel, device, launch, candidate_count = key
    return f"{device} | {candidate_count:,} candidates | {launch}"


def context_executor(key: tuple) -> str | None:
    identities, _kernel, _device, launch, _candidate_count = key
    description = " ".join([*identities, str(launch)]).casefold()
    if "cpuompblocks" in description:
        return "CpuOmpBlocks"
    if "gpucuda" in description:
        return "GpuCuda"
    return None


def context_label(example: str, key: tuple) -> str:
    """Return the human-relevant part of a context identity."""
    labels = []
    for label in context_name(key).split(", "):
        if label.lower().endswith("cuda"):
            continue
        if label == example:
            continue
        if label.startswith(f"{example}/"):
            label = label[len(example) + 1 :]
        labels.append(label)
    return " / ".join(labels) or "main kernel"


def collect(root: Path) -> dict[str, dict[tuple, dict[str, dict]]]:
    examples: dict[str, dict[tuple, dict[str, dict]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for history_path in sorted(root.glob("*/*/complete-history.json")):
        example = history_path.parents[1].name
        strategy = history_path.parent.name
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exception:
            print(f"Skipping {history_path}: {exception}", file=sys.stderr)
            continue
        for cache in history.get("contexts", {}).values():
            improvements = cache.get("best_improvements", [])
            has_measurements = any(cache.get("candidate_samples", []))
            if improvements or has_measurements:
                cache["best_improvements"] = sorted(
                    improvements, key=lambda item: item["execution_count"]
                )
                examples[example][context_key(cache)][strategy] = cache
    return examples


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def improvement_series(improvements: list[dict]) -> tuple[list[int], list[float]]:
    return (
        [item["execution_count"] for item in improvements],
        [item["runtime_seconds"] * 1.0e6 for item in improvements],
    )


def candidate_series(cache: dict) -> tuple[list[int], list[float]]:
    """Return every measured candidate and its robust runtime estimate."""
    indexes: list[int] = []
    runtimes: list[float] = []
    samples = cache.get("candidate_samples", [])
    estimates = cache.get("candidate_estimates", [])
    for index, candidate_samples in enumerate(samples):
        estimate = estimates[index] if index < len(estimates) else None
        if estimate is None and candidate_samples:
            estimate = statistics.median(candidate_samples)
        if estimate is not None:
            indexes.append(index)
            runtimes.append(estimate * 1.0e6)
    return indexes, runtimes


def load_baseline(path: Path | None) -> dict[str, dict[str, tuple[float, str]]]:
    if path is None:
        return {}
    summary_path = path / "summary.json" if path.is_dir() else path
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exception:
        raise ValueError(f"cannot load baseline summary {summary_path}: {exception}") from exception
    baselines: dict[str, dict[str, tuple[float, str]]] = {}
    for example, values in summary.items():
        if not isinstance(values, dict):
            continue
        runtime_kind = values.get("reported_runtime_kind", "runtime")
        reported = values.get("reported_runtimes")
        if isinstance(reported, dict):
            for executor, executor_values in reported.items():
                if not isinstance(executor_values, dict):
                    continue
                runtime = executor_values.get("mean_runtime_seconds")
                if isinstance(runtime, (int, float)):
                    baselines.setdefault(example, {})[executor] = (
                        runtime,
                        runtime_kind,
                    )
        # Backward compatibility for CUDA-only summaries written by the old
        # baseline runner.
        runtime = values.get("mean_reported_cuda_runtime_seconds")
        if isinstance(runtime, (int, float)):
            baselines.setdefault(example, {}).setdefault(
                "GpuCuda", (runtime, runtime_kind)
            )
    return baselines


def best_candidates(strategy_caches: dict[str, dict], limit: int = 5) -> list[dict]:
    """Return the fastest unique measured candidates across all strategies."""
    candidates: dict[int, dict] = {}
    for strategy, cache in strategy_caches.items():
        samples = cache.get("candidate_samples", [])
        estimates = cache.get("candidate_estimates", [])
        configurations = cache.get("candidate_configurations", [])
        for index, candidate_samples in enumerate(samples):
            estimate = estimates[index] if index < len(estimates) else None
            if estimate is None and candidate_samples:
                estimate = statistics.median(candidate_samples)
            if estimate is None:
                continue
            candidate = {
                "candidate_index": index,
                "runtime_seconds": estimate,
                "strategy": strategy,
                "configuration": configurations[index] if index < len(configurations) else {},
            }
            previous = candidates.get(index)
            if previous is None or candidate["runtime_seconds"] < previous["runtime_seconds"]:
                candidates[index] = candidate
    return sorted(candidates.values(), key=lambda item: item["runtime_seconds"])[:limit]


def format_best_candidates(candidates: list[dict]) -> str:
    if not candidates:
        return "Five fastest measured candidates: unavailable in this history"
    lines = ["Five fastest measured candidates"]
    for rank, candidate in enumerate(candidates, 1):
        configuration = ", ".join(
            f"{name}={value}" for name, value in candidate["configuration"].items()
        )
        if not configuration:
            configuration = f"candidate={candidate['candidate_index']}"
        strategy = STRATEGY_LABELS.get(candidate["strategy"], candidate["strategy"])
        prefix = f"{rank}. {candidate['runtime_seconds'] * 1.0e6:.3f} µs [{strategy}] "
        lines.extend(
            textwrap.wrap(
                configuration,
                width=82,
                initial_indent=prefix,
                subsequent_indent=" " * len(prefix),
            )
        )
    return "\n".join(lines)


def plot_context(
    axis: plt.Axes,
    example: str,
    key: tuple,
    strategy_caches: dict[str, dict],
    *,
    view: str,
    show_details: bool,
    baseline: dict[str, tuple[float, str]] | None = None,
) -> None:
    for strategy, cache in sorted(strategy_caches.items()):
        label = STRATEGY_LABELS.get(strategy, strategy)
        if view == "candidates":
            candidate_indexes, runtime_microseconds = candidate_series(cache)
            if candidate_indexes:
                axis.scatter(
                    candidate_indexes,
                    runtime_microseconds,
                    s=5,
                    alpha=0.45,
                    linewidths=0,
                    label=label,
                )
        else:
            launches, runtime_microseconds = improvement_series(cache["best_improvements"])
            if not launches:
                continue
            axis.plot(
                launches,
                runtime_microseconds,
                marker="o",
                markersize=3,
                linewidth=1.25,
                label=label,
            )
    if baseline:
        executor = context_executor(key)
        selected = (
            {executor: baseline[executor]}
            if executor is not None and executor in baseline
            else baseline
        )
        for baseline_executor, (runtime, kind) in sorted(selected.items()):
            kind_label = "kernel" if kind == "kernel" else "time step"
            style = BASELINE_STYLES.get(
                baseline_executor, {"color": "black", "linestyle": "--"}
            )
            axis.axhline(
                runtime * 1.0e6,
                linewidth=1.3,
                label=(
                    f"Untuned Alpaka {baseline_executor} mean {kind_label}"
                ),
                **style,
            )
    axis.set_title(context_label(example, key), fontsize=10, loc="left")
    if show_details:
        _identities, _kernel, device, _launch, candidate_count = key
        axis.text(
            0.99,
            0.98,
            f"{device} · {candidate_count:,} candidates",
            transform=axis.transAxes,
            va="top",
            ha="right",
            fontsize=7,
            color="0.35",
        )
    if view == "candidates":
        axis.set_xlabel("Candidate index")
        axis.set_ylabel("Measured robust runtime [µs]")
    else:
        axis.set_xlabel("Tuning launches")
        axis.set_ylabel("Best robust runtime [µs]")
    axis.grid(True, alpha=0.25)


def add_legend(figure: plt.Figure, axes: object) -> None:
    handles: dict[str, object] = {}
    for axis in axes.flat:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            handles[label] = handle
    if handles:
        figure.legend(
            handles.values(),
            handles.keys(),
            loc="outside upper center",
            ncols=min(3, len(handles)),
        )


def render_examples(
    data: dict,
    plot_directory: Path,
    baselines: dict[str, dict[str, tuple[float, str]]],
    view: str,
) -> list[Path]:
    written: list[Path] = []
    for example, contexts in sorted(data.items()):
        keys = sorted(contexts, key=context_name)
        columns = 1 if len(keys) < 4 else 2
        rows = math.ceil(len(keys) / columns)
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(7.5 * columns, 6.2 * rows),
            squeeze=False,
            layout="constrained",
        )
        for axis, key in zip(axes.flat, keys):
            plot_context(
                axis,
                example,
                key,
                contexts[key],
                view=view,
                show_details=True,
                baseline=baselines.get(example),
            )
            context = context_label(example, key)
            title = example if context == "main kernel" else f"{example} · {context}"
            axis.set_title(title, fontsize=10, loc="left")
        for axis in list(axes.flat)[len(keys):]:
            axis.remove()

        add_legend(figure, axes)
        prefix = "" if view == "candidates" else "best_"
        destination = plot_directory / f"{prefix}{safe_name(example)}.png"
        figure.savefig(destination, dpi=180, bbox_inches="tight")
        plt.close(figure)
        written.append(destination)
    return written


def render_overview(
    data: dict,
    plot_directory: Path,
    baselines: dict[str, dict[str, tuple[float, str]]],
    view: str,
) -> Path:
    panels = [
        (example, key, contexts[key])
        for example, contexts in sorted(data.items())
        for key in sorted(contexts, key=context_name)
    ]
    columns = min(3, max(1, len(panels)))
    rows = math.ceil(len(panels) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(6.0 * columns, 4.2 * rows),
        squeeze=False,
        layout="constrained",
    )
    for axis, (example, key, strategies) in zip(axes.flat, panels):
        plot_context(
            axis,
            example,
            key,
            strategies,
            view=view,
            show_details=False,
            baseline=baselines.get(example),
        )
        context = context_label(example, key)
        title = example if context == "main kernel" else f"{example} · {context}"
        axis.set_title(title, fontsize=9, loc="left")
    for axis in list(axes.flat)[len(panels):]:
        axis.remove()
    add_legend(figure, axes)
    prefix = "" if view == "candidates" else "best_"
    destination = plot_directory / f"{prefix}overview.png"
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination


def write_dashboard(plot_directory: Path, examples: list[Path], overview: Path) -> Path:
    buttons = "\n".join(
        f'<button data-plot="{html.escape(path.name)}">{html.escape(path.stem)}</button>'
        for path in examples
    )
    destination = plot_directory / "index.html"
    destination.write_text(
        f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>alpakaTune benchmark plots</title>
<style>
body {{ margin: 0; font-family: system-ui, sans-serif; background: #f5f6f8; color: #20242a; }}
header {{ position: sticky; top: 0; z-index: 1; padding: .8rem 1rem; background: #fff; box-shadow: 0 1px 5px #0002; }}
button {{ margin: .2rem; padding: .45rem .75rem; border: 1px solid #bbc2ca; border-radius: .4rem; background: #fff; cursor: pointer; }}
button.active {{ color: #fff; background: #3465a4; border-color: #3465a4; }}
main {{ padding: 1rem; text-align: center; }}
img {{ max-width: 100%; height: auto; background: #fff; box-shadow: 0 1px 6px #0002; }}
</style>
<header>
  <div id="views">
    <button class="active" data-view="candidates">All measured candidates</button>
    <button data-view="best">Best-so-far</button>
  </div>
  <div id="plots">
    <button class="active" data-plot="{html.escape(overview.name)}">Overview</button>
    {buttons}
  </div>
</header>
<main><img id="plot" src="{html.escape(overview.name)}" alt="Benchmark plot"></main>
<script>
const image = document.querySelector('#plot');
let view = 'candidates';
let plot = '{html.escape(overview.name)}';
function updateImage() {{
  image.src = view === 'best' ? `best_${{plot}}` : plot;
}}
for (const button of document.querySelectorAll('[data-view]')) {{
  button.addEventListener('click', () => {{
    view = button.dataset.view;
    for (const other of document.querySelectorAll('[data-view]')) other.classList.remove('active');
    button.classList.add('active');
    updateImage();
  }});
}}
for (const button of document.querySelectorAll('[data-plot]')) {{
  button.addEventListener('click', () => {{
    plot = button.dataset.plot;
    for (const other of document.querySelectorAll('[data-plot]')) other.classList.remove('active');
    button.classList.add('active');
    updateImage();
  }});
}}
</script>
</html>
""",
        encoding="utf-8",
    )
    return destination


def render(
    root: Path,
    output: Path | None = None,
    baseline: Path | None = None,
) -> list[Path]:
    data = collect(root)
    if not data:
        return []
    plot_directory = output or root / "plots"
    plot_directory.mkdir(parents=True, exist_ok=True)
    baselines = load_baseline(baseline)
    examples = render_examples(data, plot_directory, baselines, "candidates")
    overview = render_overview(data, plot_directory, baselines, "candidates")
    best_examples = render_examples(data, plot_directory, baselines, "best")
    best_overview = render_overview(data, plot_directory, baselines, "best")
    dashboard = write_dashboard(plot_directory, examples, overview)
    return [*examples, overview, *best_examples, best_overview, dashboard]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--baseline",
        type=Path,
        help="baseline result directory or summary.json to overlay",
    )
    arguments = parser.parse_args()
    try:
        written = render(arguments.results.resolve(), arguments.output, arguments.baseline)
    except ValueError as exception:
        print(exception, file=sys.stderr)
        return 2
    if not written:
        print("No persisted candidate measurements were found.", file=sys.stderr)
        return 1
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
