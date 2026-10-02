// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0
#include "ExampleObjectives.hpp"
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
} // namespace

TEST_CASE("normalized metric objectives enforce units, scope and availability",
          "[metrics][objective]") {
  using namespace alpakaTune::example;
  ObjectiveOptions options;
  options.objective = "weighted";
  CHECK_THROWS_AS(options.validate(), std::invalid_argument);
  options.timeScale = 2.0;
  options.counterScale = 4.0;
  options.validate();
  auto result = fixtureResult(2.0, 4u);
  CHECK(scoreMetrics(result, options) == 1.0);
  CHECK(scoreMetrics(fixtureResult(1.0, 2u), options) == 0.5);
  auto const initialIdentity = options.identity();
  options.timeWeight = 0.25;
  CHECK(options.identity() != initialIdentity);
  options.timeWeight = 0.5;
  SECTION("unavailable is not zero or runtime fallback") {
    result.metrics[1].status = alpakaMetrics::MetricStatus::permissionDenied;
    result.metrics[1].value.reset();
    result.metrics[1].diagnostic = "counter permission denied";
    CHECK_THROWS_AS(scoreMetrics(result, options), std::runtime_error);
  }
  SECTION("wrong unit") {
    result.metrics[1].descriptor.unit = alpakaMetrics::MetricUnit::bytes;
    CHECK_THROWS_AS(scoreMetrics(result, options), std::runtime_error);
  }
  SECTION("device scope cannot masquerade as queue counters") {
    result.metrics[1].descriptor.scope = alpakaMetrics::MetricScope::device;
    CHECK_THROWS_AS(scoreMetrics(result, options), std::runtime_error);
  }
  SECTION("invalid input") {
    result.metrics[0].value = std::numeric_limits<double>::infinity();
    CHECK_THROWS_AS(scoreMetrics(result, options), std::runtime_error);
  }
  SECTION("single counter preserves its value") {
    options.objective = "instructions";
    CHECK(scoreMetrics(result, options) == 4.0);
  }
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
    SKIP(std::string{alpakaTune::example::statusName(counter.status)} + ": " +
         counter.diagnostic);
  alpakaTune::example::ObjectiveOptions options;
  options.objective = "instructions";
  CHECK(alpakaTune::example::scoreMetrics(result, options) > 0.0);
}
