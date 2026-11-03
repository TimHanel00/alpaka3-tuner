# Running the tuning examples

For a completed FP32 GEMM optimization case, see [compiled matmul winners](matmul/README.md).
It provides a direct library call, the original baseline, selected configurations,
and measured GPU and CPU comparisons, including strict FP32 cuBLAS.

The [heat-equation tuning showcase](heatEquation/README.md) exports compiled
stencil winners and includes CPU/GPU runtime and hierarchical roofline comparisons.

alpakaTune is a library inside these applications; it does not own their main
loop. Every application decides independently how often its kernel is needed.
The examples use the following teaching pattern:

```cpp
auto tuner = alpakaTune::makeTuner(
    alpakaTune::TunableBundle{alpakaTune::tuneNumFrames(
        alpakaTune::generate::linSpace(Vec{1u}, Vec{64u}, Vec{1u}))},
    device, "my-kernel"); // restrict the tuning context to this device

constexpr std::size_t minimumExecutions = 50'000u;
for (std::size_t runs = 0u; runs < minimumExecutions || !tuner.completed(); ++runs)
    tuner.enqueue(queue, frameSpec, Kernel{}, args...);
```

Every example follows this shape: the tunables are written inline in
`makeTuner`, the remaining arguments restrict the tuning context, and the kernel
and its arguments are passed to `enqueue` exactly as to an alpaka queue.

This loop runs at least 50,000 launches and continues until the tuning-policy
goal is reached. In your application, use the loop required by your workload:
if it needs exactly `N` launches, run `N` iterations. Checking `completed()`
is optional.

The shipped configuration uses online exploration and adaptive selection,
with `maximum_executions: 40000` and `horizon: 40000`. The horizon schedules
admission and cooling; the execution limit ends exploration. `completed()` then
becomes true while adaptive selection keeps measuring known configurations.
The final 10,000 example launches demonstrate incumbent selection and periodic
probes without new search. Fixed selection instead locks its winner.

Offline exploration reports completion once compatible measurements are loaded.
It can use fixed replay or adaptive measurement of known configurations. The
application still owns its lifetime and should keep running as long as required.

`vectorAdd`, `grayScale`, and `matrixMultiplication` expose their application
minimum through their existing run-count command-line options. The heat-equation
and n-body examples keep their normal scientific time-step behavior unless an
explicit tuning-collection option is selected.
