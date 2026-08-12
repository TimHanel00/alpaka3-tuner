// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>
#include <alpakaTune/alpakaTune.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cstddef>
#include <string>

namespace {

inline constexpr auto runtimeValue = ALPAKA_TUNE_TUNABLE("runtimeValue");

struct WriteKernel {
  ALPAKA_FN_ACC void operator()(auto const &acc,
                                alpaka::concepts::IMdSpan auto output,
                                int value) const {
    static_cast<void>(acc);
    output[0u] = value;
  }
};

} // namespace

TEST_CASE("an active strategy recommendation is accepted", "[scheduler]") {
  auto const launchCount = GENERATE(1u, 2u, 3u);
  CAPTURE(launchCount);

  using Index = alpaka::Vec<std::size_t, 1u>;
  auto const backend =
      alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
  auto selector = alpaka::onHost::makeDeviceSelector(backend);
  if (!selector.isAvailable())
    SKIP("The host backend is unavailable.");
  auto device = selector.makeDevice(0u);
  auto queue = alpakaTune::makeQueue(device, alpaka::queueKind::nonBlocking,
                                     alpakaTune::timing::enabled);
  auto const frameSpec =
      alpaka::onHost::FrameSpec{Index{1u}, Index{1u}, alpaka::exec::cpuSerial};
  auto host = alpaka::onHost::allocHost<int>(Index{1u});
  auto output = alpaka::onHost::allocLike(device, host);

  auto config = alpakaTune::TunerConfig{};
  config.mode = alpakaTune::TuningMode::onlineAdaptive;
  config.queue = alpakaTune::QueueConfig{.disable = false,
                                         .warmupRuns = 0u,
                                         .noiseCancellationWindow = 50u,
                                         .maxConsecutiveRuns = 3u};
  config.strategy = alpakaTune::StrategyKind::exhaustive;
  config.runsPerCandidate = 3u;
  config.minimumRunsPerCandidate = 3u;
  config.maximumConsecutiveStrategyRetries = 1u;
  config.horizon.reset();
  config.maximumExecutions.reset();
  config.maximumRetiredConfigurations.reset();
  config.history.file.reset();
  config.history.read = false;
  config.history.write = false;
  config.completeHistory.file.reset();
  config.completeHistory.read = false;
  config.completeHistory.write = false;

  auto const tunables = alpakaTune::TunableBundle{
      alpakaTune::named(runtimeValue, alpakaTune::RVals{7})};
  auto tuner = alpakaTune::makeTuner(config, tunables, device,
                                     "accepted-active-recommendation-test-" +
                                         std::to_string(launchCount));
  auto const bundle = alpaka::KernelBundle{
      WriteKernel{}, output.getMdSpan(), alpakaTune::markTunable(runtimeValue)};

  for (std::size_t launch = 0u; launch < launchCount; ++launch) {
    auto const observation = tuner.enqueueObserved(queue, frameSpec, bundle);
    REQUIRE(observation.measured);
  }

  auto const info = tuner.info();
  CHECK(info.activeDuplicateAcceptedCount >= launchCount);
  CHECK(info.consecutiveStrategyRetries == 0u);
  CHECK(info.strategyRetryLimitReachedCount == 0u);
  CHECK(info.adaptiveRetryFallbackCount == 0u);
  CHECK(tuner.candidateRuntimeSamples(0u).size() == launchCount);
}
