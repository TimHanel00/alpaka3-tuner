# Tuning-example application lifetime

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

This means “run at least 50,000 application launches and do not stop before the
configured tuning-policy goal has been reached.” It does **not** mean that a
normal alpakaTune user must structure an application this way. An application
that needs exactly `N` kernel launches should use its own `N`-iteration loop
and may ignore `completed()` completely.

The default tuner configuration uses `maximum_executions: 40000`. The two
numbers are intentionally independent:

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
