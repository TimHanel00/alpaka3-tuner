# alpakaTune

alpakaTune is a header-only C++20 library for tuning alpaka3 kernels.
It measures candidate launch shapes and kernel parameters, then uses those
measurements to choose configurations for later launches.

To tune an existing Alpaka launch, keep your `device`, `FrameSpec` named
`frame`, and `KernelBundle` named `bundle`. Create one tuner, then replace
`queue.enqueue(frame, bundle)` in your application loop:

```cpp
#include <alpaka/alpaka.hpp>
#include <alpakaTune/alpakaTune.hpp>
#include <cstddef>

auto queue = device.makeQueue(
    alpaka::queueKind::nonBlocking, alpaka::timing::enabled);
auto launchChoices = alpakaTune::makeAutomaticLaunchTuning(frame);
auto tuner = alpakaTune::makeTuner(
    alpakaTune::TunerConfig{}, alpakaTune::TunableBundle{launchChoices}, device);

for (std::size_t step = 0u; step < 100u; ++step)
    tuner.enqueue(queue, frame, bundle);

alpaka::onHost::wait(queue);
```

Each call launches your kernel once while preserving its logical coverage.
Keep fills and copies on the same queue, and wait before reading results on the host.

Start with [Your first tuner](https://alpaka3-tuner.readthedocs.io/en/latest/getting_started.html)
for a complete example and setup instructions. The
[documentation](https://alpaka3-tuner.readthedocs.io/en/latest/) covers launch tuning,
saved history, and custom objectives; [examples](example/README.md) show larger applications.
