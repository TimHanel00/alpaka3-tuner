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
# Select any enabled backend/executor using the shared example options.
build/example/adaptivePi/alpakaTune_adaptivePi --backend cuda:nvidiaGpu --objective accuracy
```

Configure with `-DalpakaTune_BUILD_EXAMPLES=ON`. Enable the desired Alpaka backend at configuration time. By default the example
runs each enabled device/executor pairing with available hardware. Use
`--backend api:deviceKind` and/or `--executor name` to select a pairing.

## Algorithm

Start with the square `[0,1] × [0,1]`. At each level, an Alpaka kernel evaluates
the active frontier on the device:

1. Retain the whole area when all four corners are inside the circle. The
   circle is convex, so the whole square is inside.
2. Discard the whole tile when its lower-left corner is outside or on the
   circle. In this quadrant that corner is the nearest point to the origin.
3. Subdivide unresolved boundary tiles into four children until the selected
   maximum depth is reached.
4. At that depth only, sample random points in each remaining boundary tile.

Known interior tiles write their full area into a 2D contribution buffer.
Final boundary tiles are block work items: threads generate samples, reduce
hits in shared memory, and write one value weighted by that tile’s area.
`onHost::reduce` sums this 2D buffer, then the host multiplies the area by four.
Only the final scalar returns to the host in normal execution.
Inside/outside tiles never re-enter the frontier. Monte Carlo points are not
spent on early levels. Alpaka’s Philox engine uses strided sample indices as seeds, and its uniform
real distribution samples coordinates inside the tile.

The quarter-circle hit-count idea loosely follows Alpaka's
[Monte Carlo integration example](https://github.com/alpaka-group/alpaka/blob/develop/example/monteCarloIntegration/src/monteCarloIntegration.cpp).
Adaptive refinement, tile retirement, and the optimization scheme are separate.

```text
host tuning kernel(split count, samples per tile)
  refine active masks → select boundary tiles → sample/reduce each tile by block
  → onHost::reduce(tileContributions) → Pi
```

## Two parameters and 320 configurations

| Parameter | Candidates |
| --- | --- |
| Maximum split depth | 2, 4, 6, 8 |
| Samples per final boundary tile | 32, 64, …, 2560 (multiples of 32) |

The tuner runs a host kernel with these two arguments. That kernel launches the
refinement, final sampling and reduction kernels on the calculation queue.
Their FrameSpecs stay fixed at 16 × 8 frames with an 8 × 16 frame extent. Four split depths
and 80 sample counts give 320 configurations.

## Objective and constraint

Accuracy mode constructs the tuner with `customMetric("absolute_pi_error")` and
uses a timing-disabled queue. After each calculation it calls:

```cpp
tuner.provideMetric(std::abs(result.estimate - std::numbers::pi));
tuner.lastConfig().valid = result.seconds <= runtimeBudget;
```

Runtime mode constructs the tuner without a custom-metric token and uses a
timing-enabled queue. Default kernel timing covers the host orchestration and
all inner phases, since the calculator waits for the final reduction. Afterward
it calls:

```cpp
tuner.lastConfig().valid =
    std::abs(result.estimate - std::numbers::pi) <= maximumError;
```

The timer includes active-mask updates, transfers, classification, sampling,
and area reduction. Reusable buffer allocation occurs before calculations.
The runtime budget is a **post-evaluation acceptance limit**, not a deadline
that interrupts a running kernel. The search and verification perform many
calculations; 10 ms constrains one calculation, not the complete search.

The integration tests validate tile-area conservation, geometric lower/upper Pi bounds,
final-depth-only sampling, feasible winner selection, and four terminal replays.
One measurement per candidate keeps this feature showcase short;
increase samples per candidate for production runtime comparisons.

The same code is exercised by `alpakaTune_adaptive_pi_tests`, including zero
runtime and zero-error constraints that reject every candidate without inventing
a fallback winner. See [custom tuning metrics](../../docs/source/custom_metrics.rst)
and [advanced objective examples](../../docs/source/advanced_objectives.rst).

## Source layout

- `PiKernel.hpp`: tile refinement, block sampling and shared-memory reduction.
- `PiCalculator.hpp`: buffers, refinement launches and `onHost::reduce`.
- `AdaptivePi.hpp`: the two tuning parameters, objective and postcondition.
- `adaptivePi.cpp`: arguments, enabled backend selection and result reporting.

Refinement and final-level collection are separate kernel instantiations; the
sampling kernel contains no tile-classification branches. The device FrameSpecs
stay fixed, and timing includes the complete pipeline.

Active tiles are represented by two alternating 2D masks. Refinement writes
children to fixed positions in the next mask. Inactive slots are traversed but
skipped; they neither subdivide nor generate samples. Retired tiles stay inactive;
there are no append counters or explicit global atomics in these kernels.
The leaf-description buffer is available for inspection. Integration checks
read it outside the timed calculation; normal example runs transfer only Pi.
