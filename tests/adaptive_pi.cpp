// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0
#include "AdaptivePi.hpp"
#include <catch2/catch_test_macros.hpp>

namespace piTests {
void checkOptimization(alpaka::onHost::concepts::Device auto device,
                       alpaka::concepts::Executor auto executor) {
  auto result = adaptivePi::optimize(device, executor);
  REQUIRE(result.candidateCount == 320u);
  REQUIRE(result.feasible);
  CHECK(result.replayCount == 4u);
  CHECK(result.winner.seconds <= 0.01);
  CHECK(result.bestError < 0.001);
  CHECK(result.parameters == result.winner.parameters);
  CHECK(result.parameters[0u] % 2u == 0u);
  auto fastest = adaptivePi::optimize(device, executor, 0.01,
                                      adaptivePi::Objective::runtime);
  REQUIRE(fastest.feasible);
  CHECK(fastest.candidateCount == 320u);
  CHECK(fastest.bestError <= 1.0e-3);
  CHECK(fastest.bestMetric > 0.0);
  CHECK(fastest.replayCount == 4u);
  CHECK(fastest.rejectedCount > 0u);
  CHECK(result.parameters[1u] % 32u == 0u);
  CHECK(result.parameters[3u] % 32u == 0u);
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
#if ALPAKA_LANG_CUDA && defined(ALPAKA_CMAKE_TARGET_CUDA)
TEST_CASE("adaptive Pi optimizes a parallel CUDA tile calculation",
          "[pi][integration][cuda]") {
  auto selector = alpaka::onHost::makeDeviceSelector(alpaka::onHost::DeviceSpec{
      alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu});
  if (!selector.isAvailable())
    SKIP("A CUDA device is required for Pi GPU execution.");
  piTests::checkOptimization(selector.makeDevice(0u), alpaka::exec::gpuCuda);
}
#endif

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
