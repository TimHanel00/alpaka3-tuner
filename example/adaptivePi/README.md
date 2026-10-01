# Adaptive Pi: swap the objective and the post-evaluation constraint

This example uses one calculator, one search space, and one `main()` for two
objectives. It requires no new tuner interface.

```sh
cmake --build build --target alpakaTune_adaptivePi
# Minimize absolute Pi error; each complete calculation must finish in 10 ms.
build/example/adaptivePi/alpakaTune_adaptivePi --objective accuracy --runtime-budget-ms 10
# Minimize the default synchronized runtime; absolute Pi error must be <= 1e-3.
build/example/adaptivePi/alpakaTune_adaptivePi --objective runtime --max-error 1e-3
# The limits are configurable; these are the defaults.
build/example/adaptivePi/alpakaTune_adaptivePi --objective runtime --max-error 5e-4
# A CUDA build can run the same tile calculator on the GPU.
build/example/adaptivePi/alpakaTune_adaptivePi --cuda --objective accuracy
```

Configure with `-DalpakaTune_BUILD_EXAMPLES=ON`. Enable `-Dalpaka_DEP_CUDA=ON`
for `--cuda`. The host calculator remains available in a CUDA build.

## Algorithm

Start with the square `[0,1] × [0,1]`. At each level, an Alpaka kernel evaluates
all remaining tiles in parallel:

1. Retain the whole area when all four corners are inside the circle. The
   circle is convex, so the whole square is inside.
2. Discard the whole tile when its lower-left corner is outside or on the
   circle. In this quadrant that corner is the nearest point to the origin.
3. Subdivide unresolved boundary tiles into four children until the selected
   maximum depth is reached.
4. At that depth only, sample random points in each remaining boundary tile.

Sum exact interior areas and sampled boundary areas, then multiply by four.
Inside/outside tiles never re-enter the frontier. Monte Carlo points are not
spent on early levels. Counter-based random streams are deterministic per tile,
so worker order and frame geometry do not change the numerical result.

The quarter-circle hit-count idea loosely follows Alpaka's
[Monte Carlo integration example](https://github.com/alpaka-group/alpaka/blob/develop/example/monteCarloIntegration/src/monteCarloIntegration.cpp).
Adaptive refinement, tile retirement, and the optimization scheme are separate.

## Four parameters and 320 configurations

| Parameter | Candidates |
| --- | --- |
| Maximum split depth | 2, 4, 6, 8 |
| Monte Carlo points per final boundary tile | 32, 64, 128, 256, 512 |
| Inner FrameSpec `numFrames` | 1, 2, 4, 8 |
| Inner FrameSpec `frameExtent` | 32, 64, 128, 256 |

The Cartesian product has **320** candidates. All inner refinement launches use
the selected geometry and grid-stride tile traversal. Frame geometry controls
work distribution; it does not constrain the number of tiles processed.

A once-only host kernel orchestrates the complete calculation, including inner
GPU launches when requested. Its outer FrameSpec is fixed at one frame and one
worker. The two geometry tunables are ordinary named values used to construct
**inner** FrameSpecs; applying them to the outer orchestration would execute the
entire calculation multiple times.

## Objective and constraint

Accuracy mode constructs the tuner with `customMetric("absolute_pi_error")` and
uses a timing-disabled queue. After each calculation it calls:

```cpp
tuner.provideMetric(std::abs(result.estimate - std::numbers::pi));
tuner.lastConfig().valid = result.seconds <= runtimeBudget;
```

Runtime mode constructs the tuner without a custom-metric token and uses a
timing-enabled queue. Default kernel timing covers the host orchestration and
all inner phases, since the calculator waits for each phase. Afterward it calls:

```cpp
tuner.lastConfig().valid =
    std::abs(result.estimate - std::numbers::pi) <= maximumError;
```

The timer includes frontier construction, transfers, classification, sampling,
and area reduction. Reusable buffer allocation occurs before calculations.
The runtime budget is a **post-evaluation acceptance limit**, not a deadline
that interrupts a running kernel. The search and verification perform many
calculations; 10 ms constrains one calculation, not the complete search.

The example validates tile-area conservation, geometric lower/upper Pi bounds,
final-depth-only sampling, feasible winner selection, and four terminal replays.
One deterministic measurement per candidate keeps this feature showcase short;
increase samples per candidate for production runtime comparisons.

The same code is exercised by `alpakaTune_adaptive_pi_tests`, including zero
runtime and zero-error constraints that reject every candidate without inventing
a fallback winner. See [custom tuning metrics](../../docs/source/custom_metrics.rst)
and [advanced objective examples](../../docs/source/advanced_objectives.rst).
