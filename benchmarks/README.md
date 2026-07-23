# Tuning strategy benchmark

The benchmark runs every example containing an alpakaTune context with the
four model-free strategies by default. The application owns the launch count;
the examples own their 50000-launch default. The runner independently supplies
an isolated tuner configuration, policy guards, and persistent history for
each example/strategy pair.

Build the examples, install the Python dependencies, and start a run:

```bash
cmake --build build -j
python3 -m pip install -r benchmarks/requirements.txt
python3 benchmarks/run.py
```

Examples run every enabled backend and executor by default. Restrict a run to
one GPU or CPU backend and executor when desired:

```bash
python3 benchmarks/run.py --backend cuda:nvidiaGpu --executor gpuCuda
python3 benchmarks/run.py --backend host:cpu --executor cpuOmpBlocks
```

The same options are accepted by each tuned example executable directly.

Select only specific benchmark examples with `--examples`:

```bash
python3 benchmarks/run.py --examples scan vectorAdd tutorial_05_kernel
```

Or start from the selected/default list and remove examples with
`--exclude-examples`:

```bash
python3 benchmarks/run.py --exclude-examples nBody grayScale
```

When both options are present, exclusions are applied after inclusions.

The learned strategy is opt-in because it must never silently use its random
fallback during a model comparison. Supply a readable model artifact whenever
`learned_hybrid` is selected:

```bash
python3 benchmarks/run.py \
  --strategies exhaustive random simulated_annealing bayesian_optimization learned_hybrid \
  --model /absolute/path/to/model.atml \
  --learned-candidate-pool-size 4096 \
  --learned-candidate-batch-size 256 \
  --backend cuda:nvidiaGpu --executor gpuCuda
```

The runner resolves the model to an absolute path and injects `learning.model`
only into the learned pair's generated YAML. `benchmark.json` and the learned
pair's `run.json` record its SHA-256 provenance digest and the runtime-compatible
64-bit FNV-1a digest. A learned pair succeeds only when every persisted context
reports `learning.status: active`, `artifact_load_status: available`, and the
expected runtime digest. A missing or fallback model therefore fails the run.
The optional pool and batch flags are written only into learned configurations;
they make small validation pools possible without changing production defaults.

By default results are written below
`benchmarks/results/<UTC-run-id>/<example>/<strategy>/`. Every pair contains
the generated tuner configuration, persistent history, stdout/stderr logs, and `run.json`
timing/status metadata. The configured launch and retired-configuration limits
default to 40000 and 100000 respectively; they are tuner-policy limits and can
be changed with the corresponding command-line options. Separately, the
example applications default to at least 50000 launches.

For a bounded comparison, add `--tune-until-terminal`. The option name is kept
for command-line compatibility. The finite
heatEquation2D and nBody examples then keep executing their safe kernel sequence
through their 50000-launch application minimum and until ``completed()`` reports
that every tuner policy reached its goal, rather than stopping after their
normal scientific step count. With ``maximum_executions: 40000``, this records
a deliberate 10000-launch post-horizon interval. Adaptive tuning stays active
during it; a fixed tuner replays its current best configuration:

```bash
python3 benchmarks/run.py \
  --tune-until-terminal \
  --maximum-executions 40000 \
  --examples heatEquation2D nBody \
  --backend cuda:nvidiaGpu --executor gpuCuda
```

This bounded mode is separate from, and cannot be combined with,
`--full-coverage`.

Those limits count tuner activity, not distinct candidates. In particular,
`maximum_executions` includes warm-up launches and repeated timing samples. An
exhaustive run that reaches this limit may therefore retire only part of the
space even though the strategy visits candidates sequentially. `run.json`
contains per-context `history_diagnostics`, and the runner prints a warning
with the measured/legal candidate ratio whenever exhaustive collection is
capped or an example exits before its tuner completes.

## Full-coverage training data

Use `--full-coverage` to collect an unbiased, complete exhaustive runtime
surface for model training:

```bash
python3 benchmarks/run.py \
  --full-coverage \
  --examples boundaryIter grayScale heatEquation2D matrixMultiplication nBody vectorAdd \
  --backend cuda:nvidiaGpu \
  --executor gpuCuda \
  --no-plot
```

This mode runs only the exhaustive strategy, selects ``online_fixed``, removes
the retired-configuration limit, uses a tuner safety guard of one million
launches by default, disables Mann-Whitney early retirement, and records exactly three
measured launches after one warm-up for each legal
candidate residency. Override the guard with `--maximum-executions`; if it
is too short, coverage validation fails cleanly so the collection can be
resubmitted with a larger value. Full coverage cannot be combined with
`--maximum-retired-configurations` or a non-exhaustive strategy.

The finite heat-equation and n-body simulations receive the benchmark-only
`--tune-until-complete` option automatically. They continue their normal safe
kernel sequence after the configured scientific time steps until every tuning
context completes; their default command-line behavior is unchanged. The
option is also available directly on those two executables for collector
workflows.

A pair is marked completed only if every persisted context reports
`all_configurations` and the retired plus restriction-rejected candidates equal
the complete Cartesian space. This verification is stored as
`full_coverage_verified` in `run.json`, making the same runner suitable for a
local shell or a scheduler job without weakening the dataset acceptance rule.

There is intentionally no timeout. Failures are recorded and the remaining
pairs continue. Resume an interrupted result directory without repeating
successful pairs:

```bash
python3 benchmarks/run.py --output benchmarks/results/<run-id> --resume
```

The runner invokes the visualizer after the final pair. It writes one PNG per
example, a combined subplot overview, and `plots/index.html`, which provides
buttons for switching between the overview and individual examples. Plot titles
use the context identity; device, candidate count, and launch details are shown
separately in smaller text. A point is added only when a retired configuration
establishes a new lowest robust runtime: x is the tuning launch count at that
event, y is that runtime, and the line/legend denotes the strategy. Each detailed
subplot also lists the five fastest unique measured candidates, including their
strategy and tuning-parameter values. Existing results can be plotted again with:

```bash
python3 benchmarks/visualize.py benchmarks/results/<run-id>
```

## Untuned Alpaka baseline

The pinned Alpaka source tree contains the original examples without an
alpakaTune context. Configure them in a separate build directory so their target
names and measurements cannot mix with the tuned variants:

```bash
cmake -S build/_deps/alpaka3-src -B build/alpaka-baseline \
  -Dalpaka_EXAMPLES=ON \
  -Dalpaka_DEP_CUDA=ON \
  -Dalpaka_EXEC_CpuOmpBlocks=ON \
  -Dalpaka_EXEC_CpuSerial=OFF \
  -Dalpaka_EXEC_GpuCuda=ON
cmake --build build/alpaka-baseline -j
python3 benchmarks/run_baseline.py \
  --examples boundaryIter grayScale heatEquation2D nBody vectorAdd \
  --repetitions 10
```

Results are written below `benchmarks/baseline-results/<UTC-run-id>/`. Every
execution has separate stdout, stderr, and `run.json` files; `summary.json`
contains median, minimum, and maximum wall times as well as the mean
kernel/time-step runtime reported by the upstream example. It stores
`reported_runtimes` separately for `CpuOmpBlocks` and `GpuCuda`. CpuSerial must
remain compiled out: heatEquation2D and nBody print only `Host`, which is mapped
to CpuOmpBlocks under that build contract. The first reported value for each
executor is retained as a cold-start warmup measurement and excluded from its
steady-state arithmetic mean. The legacy CUDA-only summary fields remain for
older tooling. Overlay those means
on the tuning plots after both result trees are available locally:

```bash
python3 benchmarks/visualize.py benchmarks/results/<tuned-run-id> \
  --baseline benchmarks/baseline-results/<baseline-run-id>
```

The overlay uses separately labelled horizontal lines and selects the baseline
matching the tuned context's CpuOmpBlocks or GpuCuda executor. Examples
that do not report a comparable executor runtime are left without a baseline line.
The unmodified Alpaka examples
choose their enabled backends from their CMake configuration and do not support
the tuner's `--backend` or `--executor` switches. Alpaka has no upstream
`matrixMultiplication` counterpart at the pinned revision, so that example is
not included in the baseline runner.
