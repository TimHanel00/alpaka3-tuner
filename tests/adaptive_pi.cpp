// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0
#include "AdaptivePi.hpp"
#include "DeviceRequirements.hpp"
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/interfaces/catch_interfaces_capture.hpp>
#include <map>
#include <string>

namespace piTests {
// Inspect device-produced leaves outside the timed calculation. These checks
// exercise weighting, retirement and final-depth-only sampling independently.
struct InspectLeaves {
  void operator()(auto &calculator, adaptivePi::Calculation &result) const {
    auto leaves = calculator.readLeaves();
    double insideArea = 0.0, outsideArea = 0.0, boundaryArea = 0.0;
    for (auto const &leaf : leaves) {
      using adaptivePi::LeafKind;
      if (leaf.kind == LeafKind::unused)
        continue;
      CHECK(leaf.area == 1.0 / (std::uint64_t{1u} << (2u * leaf.depth)));
      if (leaf.kind == LeafKind::inside) {
        insideArea += leaf.area;
        ++result.insideTiles;
        CHECK(leaf.samples == 0u);
      } else if (leaf.kind == LeafKind::outside) {
        outsideArea += leaf.area;
        ++result.outsideTiles;
        CHECK(leaf.samples == 0u);
      } else {
        boundaryArea += leaf.area;
        ++result.sampledTiles;
        result.randomPoints += leaf.samples;
        result.deepestSample = leaf.depth;
        CHECK(leaf.depth == result.parameters[0u]);
        CHECK(leaf.samples == result.parameters[1u]);
        CHECK(leaf.hits <= leaf.samples);
      }
    }
    // The runtime constraint is a policy test, not a CI hardware benchmark.
    // Keep real device calculations and their geometric/accuracy checks, but
    // supply a reproducible duration so sanitizers and SYCL startup cannot
    // change which candidates are feasible. Exercise both sides of 10 ms.
    REQUIRE(std::isfinite(result.seconds));
    REQUIRE(result.seconds > 0.0);
    result.seconds =
        result.parameters[0u] == 8u && result.parameters[1u] > 1280u ? 0.02
                                                                     : 0.005;
    result.lowerBound = 4.0 * insideArea;
    result.upperBound = 4.0 * (insideArea + boundaryArea);
    result.partitionArea = insideArea + outsideArea + boundaryArea;
    // A quadtree with L leaves has (L-1)/3 internal nodes.
    result.splitTiles =
        (result.insideTiles + result.outsideTiles + result.sampledTiles - 1u) /
        3u;
  }
};
inline void validate(adaptivePi::Calculation const &result,
                     std::uint32_t splits, std::uint32_t points) {
  if (result.partitionArea != 1.0)
    throw std::runtime_error{
        "Pi tiles do not partition the unit square: area=" +
        std::to_string(result.partitionArea) +
        " splits=" + std::to_string(result.parameters[0u]) +
        " points=" + std::to_string(result.parameters[1u]) +
        " inside=" + std::to_string(result.insideTiles) +
        " outside=" + std::to_string(result.outsideTiles) +
        " sampled=" + std::to_string(result.sampledTiles)};
  adaptivePi::require(result.lowerBound <= std::numbers::pi &&
                          std::numbers::pi <= result.upperBound,
                      "Geometric Pi bounds exclude the reference");
  adaptivePi::require(std::isfinite(result.estimate) &&
                          result.lowerBound <= result.estimate &&
                          result.estimate <= result.upperBound,
                      "Pi estimate violates tile bounds");
  adaptivePi::require(
      result.insideTiles > 0u && result.outsideTiles > 0u &&
          result.splitTiles > 0u && result.sampledTiles > 0u,
      "Pi calculation did not exercise all tile classifications");
  adaptivePi::require(
      result.deepestSample == splits &&
          result.randomPoints == result.sampledTiles * points,
      "Pi calculation used the wrong sampling depth or point budget");
}

void checkTrials(adaptivePi::Optimization const &outcome) {
  std::map<std::array<std::uint32_t, 2u>, double> metrics;
  for (auto const &trial : outcome.trials) {
    auto const &calculation = trial.calculation;
    validate(calculation, calculation.parameters[0u],
             calculation.parameters[1u]);
    if (!trial.valid)
      metrics[calculation.parameters] = std::numeric_limits<double>::infinity();
    else if (!trial.replay)
      metrics[calculation.parameters] = trial.metric;
  }
  double minimum = std::numeric_limits<double>::infinity();
  for (auto const &[parameters, metric] : metrics)
    minimum = std::min(minimum, metric);
  if (outcome.feasible)
    CHECK(outcome.bestMetric == minimum);
}

void checkOptimization(alpaka::onHost::concepts::Device auto device,
                       alpaka::concepts::Executor auto executor) {
  auto result = adaptivePi::optimize(
      device, executor, 0.01, adaptivePi::Objective::accuracy, 1.0e-3,
      InspectLeaves{}, Catch::getResultCapture().getCurrentTestName());
  checkTrials(result);
  REQUIRE(result.candidateCount == 320u);
  REQUIRE(result.feasible);
  CHECK(result.replayCount == 4u);
  CHECK(result.rejectedCount > 0u);
  CHECK(result.winner.seconds <= 0.01);
  CHECK(result.bestError < 0.001);
  CHECK(result.parameters == result.winner.parameters);
  CHECK(result.parameters[0u] % 2u == 0u);
  auto fastest = adaptivePi::optimize(
      device, executor, 0.01, adaptivePi::Objective::runtime, 1.0e-3,
      InspectLeaves{}, Catch::getResultCapture().getCurrentTestName());
  checkTrials(fastest);
  REQUIRE(fastest.feasible);
  CHECK(fastest.candidateCount == 320u);
  CHECK(fastest.bestError <= 1.0e-3);
  CHECK(fastest.bestMetric > 0.0);
  CHECK(fastest.replayCount == 4u);
  CHECK(fastest.rejectedCount > 0u);
  CHECK(result.parameters[1u] % 32u == 0u);
}
} // namespace piTests
TEST_CASE("adaptive Pi minimizes error within ten milliseconds",
          "[pi][integration]") {
  auto selector = alpaka::onHost::makeDeviceSelector(
      alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu});
  REQUIRE(selector.isAvailable());
  piTests::checkOptimization(selector.makeDevice(0u), alpaka::exec::cpuSerial);
}
TEST_CASE("runtime post-validation rejects every infeasible Pi configuration",
          "[pi][integration]") {
  auto selector = alpaka::onHost::makeDeviceSelector(
      alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu});
  auto result = adaptivePi::optimize(selector.makeDevice(0u),
                                     alpaka::exec::cpuSerial, 0.0);
  CHECK_FALSE(result.feasible);
  CHECK(result.candidateCount == 320u);
  CHECK(result.rejectedCount == result.candidateCount);
  CHECK(result.replayCount == 0u);
}
TEST_CASE("adaptive Pi optimizes enabled parallel backends",
          "[pi][integration][backends]") {
  std::size_t available = 0u;
  alpaka::onHost::executeForEach(
      [&](alpaka::concepts::BackendSpec auto const &backend) {
        auto selector = alpaka::onHost::makeDeviceSelector(
            alpaka::onHost::DeviceSpec{backend});
        if (!selector.isAvailable())
          return 0;
        INFO(alpaka::onHost::DeviceSpec{backend}.getApi().getName());
        INFO(alpaka::onHost::demangledName(alpaka::getExecutor(backend)));
        ++available;
        auto device = selector.makeDevice(0u);
        alpakaTune::test::runIfSupported<double>(device, [&] {
          piTests::checkOptimization(device, alpaka::getExecutor(backend));
        });
        return 0;
      },
      alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs,
                                  alpaka::exec::enabledExecutors));
  REQUIRE(available > 0u);
}

TEST_CASE("accuracy post-validation rejects every inaccurate Pi configuration",
          "[pi][integration]") {
  auto selector = alpaka::onHost::makeDeviceSelector(
      alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu});
  auto result =
      adaptivePi::optimize(selector.makeDevice(0u), alpaka::exec::cpuSerial,
                           0.01, adaptivePi::Objective::runtime, 0.0);
  CHECK_FALSE(result.feasible);
  CHECK(result.candidateCount == 320u);
  CHECK(result.rejectedCount == 320u);
  CHECK(result.replayCount == 0u);
}
