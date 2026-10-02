# Running the tuning examples

alpakaTune is a library inside these applications; it does not own their main
loop. Every application decides independently how often its kernel is needed.
The examples use the following teaching pattern:

```cpp
constexpr std::size_t minimumExecutions = 50'000u;
std::size_t completedExecutions = 0u;

while (completedExecutions < minimumExecutions || !tuner.completed()) {
    tuner.enqueue(queue, launchSpec, kernelBundle);
    ++completedExecutions;
}
```

This loop runs at least 50,000 launches and continues until the tuning-policy
goal is reached. In your application, use the loop required by your workload:
if it needs exactly `N` launches, run `N` iterations. Checking `completed()`
is optional.

The shipped `alpakaTune.yaml` uses `horizon: 40000` in adaptive mode.
That horizon controls the tuning schedule separately from the application loop:

- During the first 40,000 adaptive launches, revisit admission and temperature
  progress toward their final values.
- `completed()` becomes true at that horizon, but adaptive tuning remains
  active.
- The final 10,000 example launches expose behavior after the horizon:
  recommendations, measurements, rolling histories, revisits, and residual
  adapter updates continue at the final schedule state.
- In `online_fixed`, reaching a completion guard is terminal for tuning.
  Remaining application launches replay the selected best configuration.

In `offline`, `completed()` is true once a compatible history and its best
configuration have been loaded. In `online_fixed`, it becomes true after full
coverage, `maximum_executions`, or `maximum_retired_configurations`. In
`online_adaptive`, it is only a horizon diagnostic and never says that the
tuner has stopped.

`vectorAdd`, `grayScale`, and `matrixMultiplication` expose their application
minimum through their existing run-count command-line options. The heat-equation
and n-body examples keep their normal scientific time-step behavior unless an
explicit tuning-collection option is selected.
