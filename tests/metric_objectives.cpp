// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0
#include <alpakaTune/metrics.hpp>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <limits>

namespace {
auto fixtureResult(double seconds, std::uint64_t instructions) {
  alpakaMetrics::Result result;
  result.metrics.push_back(
      {.descriptor = {.name = "elapsed_time",
                      .unit = alpakaMetrics::MetricUnit::seconds,
                      .scope = alpakaMetrics::MetricScope::queueInterval},
       .status = alpakaMetrics::MetricStatus::available,
       .value = seconds});
  result.metrics.push_back(
      {.descriptor = {.name = "instructions",
                      .unit = alpakaMetrics::MetricUnit::count,
                      .scope = alpakaMetrics::MetricScope::queueWorkerThread},
       .status = alpakaMetrics::MetricStatus::available,
       .value = instructions});
  return result;
}
struct CountWork {
  void operator()(auto const &, std::atomic<std::uint64_t> *output) const {
    for (std::uint64_t i = 0; i < 1000u; ++i)
      output->fetch_add(i, std::memory_order_relaxed);
  }
};

auto weightedConfig() {
  auto config = alpakaTune::TunerConfig{};
  config.history = {.read = false, .write = false};
  config.completeHistory = {.read = false, .write = false};
  config.metric = alpakaTune::Metric::weighted;
  return config;
}
} // namespace

TEST_CASE("normalized metric objectives enforce units, scope and availability",
          "[metrics][objective]") {
  auto config = weightedConfig();
  CHECK_THROWS_AS(config.validate(), std::invalid_argument);
  config.weightedMetric.timeScale = 2.0;
  config.weightedMetric.counterScale = 4.0;
  config.validate();
  auto result = fixtureResult(2.0, 4u);
  CHECK(alpakaTune::metrics::score(result, config) == 1.0);
  CHECK(alpakaTune::metrics::score(fixtureResult(1.0, 2u), config) == 0.5);
  auto const initialIdentity = alpakaTune::metricIdentity(config);
  config.weightedMetric.timeWeight = 0.25;
  CHECK(alpakaTune::metricIdentity(config) != initialIdentity);
  config.weightedMetric.timeWeight = 0.5;
  SECTION("unavailable is not zero or runtime fallback") {
    result.metrics[1].status = alpakaMetrics::MetricStatus::permissionDenied;
    result.metrics[1].value.reset();
    result.metrics[1].diagnostic = "counter permission denied";
    CHECK_THROWS_AS(alpakaTune::metrics::score(result, config),
                    std::runtime_error);
  }
  SECTION("wrong unit") {
    result.metrics[1].descriptor.unit = alpakaMetrics::MetricUnit::bytes;
    CHECK_THROWS_AS(alpakaTune::metrics::score(result, config),
                    std::runtime_error);
  }
  SECTION("device scope cannot masquerade as queue counters") {
    result.metrics[1].descriptor.scope = alpakaMetrics::MetricScope::device;
    CHECK_THROWS_AS(alpakaTune::metrics::score(result, config),
                    std::runtime_error);
  }
  SECTION("invalid input") {
    result.metrics[0].value = std::numeric_limits<double>::infinity();
    CHECK_THROWS_AS(alpakaTune::metrics::score(result, config),
                    std::runtime_error);
  }
  SECTION("single counter preserves its value") {
    config.metric = alpakaTune::Metric::instructions;
    CHECK(alpakaTune::metrics::score(result, config) == 4.0);
  }
  SECTION("counter metrics cannot drive a runtime budget") {
    config.metric = alpakaTune::Metric::instructions;
    config.budget = alpakaTune::BudgetConfig{.expectedLaunches = 100u};
    CHECK_THROWS_AS(config.validate(), std::invalid_argument);
  }
}

TEST_CASE("TunerConfig::metric measures plain-queue launches",
          "[metrics][objective]") {
  auto device = alpaka::onHost::makeDeviceSelector(alpaka::api::host,
                                                   alpaka::deviceKind::cpu)
                    .makeDevice(0u);
  auto queue =
      device.makeQueue(alpaka::queueKind::nonBlocking, alpaka::timing::enabled);
  auto config = weightedConfig();
  config.metric = alpakaTune::Metric::elapsedTime;
  auto const spec = alpaka::onHost::FrameSpec{alpaka::Vec{1u}, alpaka::Vec{1u},
                                              alpaka::exec::cpuSerial};
  auto tuner = alpakaTune::makeTuner(
      config,
      alpakaTune::TunableBundle{alpakaTune::tuneNumFrames(
          alpakaTune::RVals{alpaka::Vec{1u}, alpaka::Vec{2u}})},
      device);
  CHECK(tuner.metricName() == "metric_v1;elapsed_time");
  std::atomic<std::uint64_t> output{};
  auto measuredLaunches = 0;
  for (int run = 0; run < 4; ++run) {
    tuner.enqueue(queue, spec, CountWork{}, &output);
    auto const &launch = tuner.lastConfig();
    if (launch.measured) {
      ++measuredLaunches;
      CHECK(launch.runtimeSeconds.has_value());
      CHECK(launch.metricValue == launch.runtimeSeconds);
    }
  }
  alpaka::onHost::wait(queue);
  CHECK(output > 0u);
  CHECK(measuredLaunches > 0);
  CHECK(tuner.info().measuredCandidateCount > 0u);

  auto custom = config;
  CHECK_THROWS_AS(
      alpakaTune::makeTuner(custom,
                            alpakaTune::TunableBundle{alpakaTune::tuneNumFrames(
                                alpakaTune::RVals{alpaka::Vec{1u}})},
                            device, alpakaTune::customMetric("custom")),
      std::invalid_argument);
}

TEST_CASE("real serial CPU instruction counters are optional",
          "[metrics][hardware]") {
  auto device = alpaka::onHost::makeDeviceSelector(alpaka::api::host,
                                                   alpaka::deviceKind::cpu)
                    .makeDevice(0u);
  auto queue = alpakaMetrics::makeQueue(
      device.makeQueue(alpaka::timing::enabled),
      alpakaMetrics::Config{.metrics = {alpakaMetrics::metric::elapsedTime,
                                        alpakaMetrics::metric::instructions}});
  std::atomic<std::uint64_t> output{};
  auto spec = alpaka::onHost::FrameSpec{alpaka::Vec{1u}, alpaka::Vec{1u},
                                        alpaka::exec::cpuSerial};
  auto result = queue.enqueue(spec, CountWork{}, &output).getResults();
  REQUIRE(output == 499500u);
  auto const &counter = result.getMetric("instructions");
  if (!counter.isAvailable())
    SKIP(std::string{alpakaTune::metrics::statusName(counter.status)} + ": " +
         counter.diagnostic);
  auto config = weightedConfig();
  config.metric = alpakaTune::Metric::instructions;
  CHECK(alpakaTune::metrics::score(result, config) > 0.0);
}
