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
