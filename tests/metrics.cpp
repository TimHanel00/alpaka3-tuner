// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0
#include <alpakaTune/metrics.hpp>
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <set>

namespace {
inline constexpr auto amount = ALPAKA_TUNE_TUNABLE("amount");
inline constexpr auto multiplier = ALPAKA_TUNE_TUNABLE("multiplier");
struct Work {
  void operator()(auto const &, std::atomic<std::uint64_t> *output,
                  std::uint32_t value, auto factor) const {
    output->fetch_add(value * decltype(factor)::value,
                      std::memory_order_relaxed);
  }
};
auto fixedConfig() {
  auto config = alpakaTune::TunerConfig{};
  config.mode = alpakaTune::TuningMode::onlineFixed;
  config.strategy = alpakaTune::StrategyKind::exhaustive;
  config.queue.reset();
  config.maximumExecutions = 20u;
  config.runsPerCandidate = 1u;
  config.minimumRunsPerCandidate = 1u;
  config.mannWhitneyEarlyStop = false;
  config.history.read = config.history.write = false;
  config.completeHistory.read = config.completeHistory.write = false;
  return config;
}
void exerciseLaunchSpec(
    alpaka::onHost::concepts::Device auto device,
    alpaka::onHost::concepts::ThreadOrFrameSpec auto const &spec) {
  auto queue =
      alpakaMetrics::makeQueue(device.makeQueue(alpaka::timing::enabled));
  std::atomic<std::uint64_t> output{};
  auto tunables =
      alpakaTune::TunableBundle{amount(alpakaTune::RVals{1u, 2u}),
                                multiplier(alpakaTune::CVals<1u, 2u>{})};
  auto objective = alpakaTune::customMetric(
      "queue_elapsed_v1", [](alpakaMetrics::Result const &result) {
        REQUIRE_FALSE(result.replayed);
        REQUIRE(result.passCount == 1u);
        auto const &elapsed = result.getMetric("elapsed_time");
        REQUIRE(elapsed.isAvailable());
        CHECK(elapsed.descriptor.unit == alpakaMetrics::MetricUnit::seconds);
        return elapsed.asDouble();
      });
  auto tuner =
      alpakaTune::makeTuner(fixedConfig(), tunables, device, objective);
  auto const bundle = alpaka::KernelBundle{Work{}, &output, amount, multiplier};
  std::set<std::uint64_t> ids;
  for (std::size_t i = 0; i < 4u; ++i) {
    auto launch = alpakaTune::metrics::enqueue(tuner, queue, spec, bundle);
    // Independent submission on a queue copy must not steal the tuner's result.
    auto shared = queue;
    auto noise = shared.enqueue(
        spec, alpaka::KernelBundle{Work{}, &output, 10u,
                                   std::integral_constant<unsigned, 1u>{}});
    REQUIRE(launch.getId() != noise.getId());
    CHECK(launch.getResults().measurementId == launch.getId());
    queue.clearMeasurements();
    REQUIRE(queue.getMeasurements().empty());
    alpakaTune::metrics::provideMetrics(tuner, launch);
    CHECK(tuner.lastConfig().metricValue.has_value());
    CHECK(ids.insert(launch.getId()).second);
    CHECK_THROWS_AS(alpakaTune::metrics::provideMetrics(tuner, launch),
                    std::logic_error);
  }
  alpaka::onHost::wait(queue);
  CHECK(output ==
        49u); // all four candidates once: (1+2)*(1+2), plus four noise kernels
  CHECK(tuner.completed());
  auto replay = alpakaTune::metrics::enqueue(tuner, queue, spec, bundle);
  auto latest = alpakaTune::metrics::enqueue(tuner, queue, spec, bundle);
  CHECK_THROWS_AS(alpakaTune::metrics::provideMetrics(tuner, replay),
                  std::logic_error);
  alpakaTune::metrics::provideMetrics(tuner, latest);
  CHECK_FALSE(tuner.lastConfig().measured);
  auto other =
      alpakaTune::makeTuner(fixedConfig(), tunables, device, objective);
  CHECK_THROWS_AS(alpakaTune::metrics::provideMetrics(other, latest),
                  std::logic_error);
  alpaka::onHost::wait(queue);

  // Existing default timing still forwards event markers without profiling
  // them.
  auto timed = alpakaTune::makeTuner(fixedConfig(), tunables, device);
  queue.clearMeasurements();
  timed.enqueue(queue, spec, bundle);
  CHECK(queue.getMeasurements().size() == 1u);
  CHECK(timed.lastConfig().runtimeSeconds.has_value());
  alpaka::onHost::wait(queue);
}
} // namespace

TEST_CASE(
    "metrics handles follow actual FrameSpec and ThreadSpec tuner launches",
    "[metrics]") {
  auto device = alpaka::onHost::makeDeviceSelector(alpaka::api::host,
                                                   alpaka::deviceKind::cpu)
                    .makeDevice(0u);
  auto frames = alpaka::onHost::FrameSpec{alpaka::Vec{1u}, alpaka::Vec{1u},
                                          alpaka::exec::cpuSerial};
  auto threads = alpaka::onHost::ThreadSpec{alpaka::Vec{1u}, alpaka::Vec{1u},
                                            alpaka::exec::cpuSerial};
  SECTION("FrameSpec") { exerciseLaunchSpec(device, frames); }
  SECTION("ThreadSpec") { exerciseLaunchSpec(device, threads); }
}
