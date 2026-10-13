# alpakaTune

alpakaTune is a header-only C++20 library for tuning alpaka3 kernels.
It measures candidate launch shapes and kernel parameters, then uses those
measurements to choose configurations for later launches.

This complete example runs on every available device for each enabled executor.
It runs the same increment kernel with different launch configurations as the tuner
explores frame counts and frame extents:

```cpp
#include <alpaka/alpaka.hpp>
#include <alpakaTune/alpakaTune.hpp>
#include <cstddef>
#include <cstdint>

// A basic example kernel for demonstration: increment each buffer element.
struct Increment {
  ALPAKA_FN_ACC void operator()(alpaka::onAcc::concepts::Acc auto const &acc,
                                alpaka::concepts::IMdSpan auto values) const {
    for (auto index :
         alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid,
                                   alpaka::IdxRange{values.getExtents()}))
      ++values[index];
  }
};

int main() {
  // Visit every enabled device/executor combination supported by alpaka.
  return alpaka::onHost::executeForEach(
      [](alpaka::concepts::BackendSpec auto const &backend) {
        auto selector = alpaka::onHost::makeDeviceSelector(
            alpaka::onHost::DeviceSpec{backend});
        auto executor = alpaka::getExecutor(backend);
        for (std::uint32_t index = 0u; index < selector.getDeviceCount();
             ++index) {
          auto device = selector.makeDevice(index);
          // Use a timing-enabled queue for runtime measurements.
          auto queue = device.makeQueue(alpaka::queueKind::nonBlocking,
                                        alpaka::timing::enabled);
          // Allocate one buffer on this device and initialize it to zero.
          auto values =
              alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{1024u});
          alpaka::onHost::fill(queue, values, 0u);
          // Derive the launch specification for this device and executor.
          auto frame = alpaka::onHost::getFrameSpec(device, executor,
                                                    values.getExtents());
          // Generate frame counts and extents with the same logical coverage.
          auto launchChoices = alpakaTune::makeAutomaticLaunchTuning(frame);
          // This overload loads the default YAML policy.
          auto tuner = alpakaTune::makeTuner(
              alpakaTune::TunableBundle{launchChoices}, device);
          // Bind the kernel to the buffer view.
          auto bundle = alpaka::KernelBundle{Increment{}, values.getMdSpan()};

          // Select a launch shape on each call; timings guide later choices.
          for (std::size_t step = 0u; step < 100u; ++step)
            tuner.enqueue(queue, frame, bundle);
          // Finish queued work before the buffer is destroyed.
          alpaka::onHost::wait(queue);
        }
        return 0;
      },
      alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs,
                                  alpaka::exec::enabledExecutors));
}
```

`makeAutomaticLaunchTuning(frame)` supplies choices for the number of frames and
the extent of each frame. Each `tuner.enqueue(...)` selects a launch configuration
and runs the kernel once. Across launches, the tuner explores these combinations
and reuses promising configurations based on measured runtimes.
alpaka3 maps the selected `FrameSpec` to thread blocks and threads according to
the device and executor; `makeIdxMap` covers every buffer element across these
configurations. See [alpaka3 kernel launching](https://alpaka3.readthedocs.io/en/latest/tutorial/kernel.html#launching-the-kernel)
for the mapping.

`makeTuner(tunables, device)` loads the process-default YAML configuration.
Set `ALPAKA_TUNE_CONFIG` to choose another file; see
[configuration](https://alpaka3-tuner.readthedocs.io/en/latest/configuration.html).

Follow the [setup and run instructions](https://alpaka3-tuner.readthedocs.io/en/latest/getting_started.html#run-the-readme-example)
for this example. A dedicated test builds and runs this exact snippet.
The [documentation](https://alpaka3-tuner.readthedocs.io/en/latest/)
covers launch tuning, saved history, and custom objectives;
[examples](example/README.md) show larger applications.
