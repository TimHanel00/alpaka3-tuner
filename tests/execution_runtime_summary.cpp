// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#include <alpakaTune/core/peripherals/ExecutionRuntimeSummary.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>

TEST_CASE("Execution runtime summary excludes untimed launches") {
  CHECK_FALSE(alpakaTune::detail::summarizeExecutionRuntimes({}));
  auto executions =
      std::array{alpakaTune::ExecutedConfiguration{.candidateIndex = 0u},
                 alpakaTune::ExecutedConfiguration{.candidateIndex = 1u}};
  CHECK_FALSE(alpakaTune::detail::summarizeExecutionRuntimes(executions));
  executions[1u].runtimeSeconds = 2.0;
  auto const result =
      alpakaTune::detail::summarizeExecutionRuntimes(executions);
  REQUIRE(result);
  CHECK(result->sampleCount == 1u);
  CHECK(result->minimumSeconds == 2.0);
  CHECK(result->maximumSeconds == 2.0);
  CHECK(result->medianSeconds == 2.0);
  CHECK(result->averageSeconds == 2.0);
  CHECK(result->minimumExecution.candidateIndex == 1u);
}

TEST_CASE("Execution runtime summary preserves the fastest launch parameters") {
  auto executions =
      std::array{alpakaTune::ExecutedConfiguration{.executionIndex = 0u,
                                                   .candidateIndex = 2u,
                                                   .configuration = {1.0f},
                                                   .runtimeSeconds = 9.0},
                 alpakaTune::ExecutedConfiguration{.executionIndex = 1u,
                                                   .candidateIndex = 1u,
                                                   .configuration = {0.5f},
                                                   .runtimeSeconds = 1.0},
                 alpakaTune::ExecutedConfiguration{.executionIndex = 2u,
                                                   .candidateIndex = 0u,
                                                   .configuration = {0.0f},
                                                   .runtimeSeconds = 5.0},
                 alpakaTune::ExecutedConfiguration{.executionIndex = 3u,
                                                   .candidateIndex = 0u,
                                                   .configuration = {0.0f},
                                                   .runtimeSeconds = 1.0}};
  // Raw execution statistics still include launches invalidated afterward.
  executions[1u].valid = false;
  auto const odd = alpakaTune::detail::summarizeExecutionRuntimes(
      std::span{executions}.first(3u));
  REQUIRE(odd);
  CHECK(odd->sampleCount == 3u);
  CHECK(odd->minimumSeconds == 1.0);
  CHECK(odd->maximumSeconds == 9.0);
  CHECK(odd->medianSeconds == 5.0);
  CHECK(odd->averageSeconds == 5.0);
  auto const even = alpakaTune::detail::summarizeExecutionRuntimes(executions);
  REQUIRE(even);
  CHECK(even->sampleCount == 4u);
  CHECK(even->minimumSeconds == 1.0);
  CHECK(even->maximumSeconds == 9.0);
  CHECK(even->medianSeconds == 3.0);
  CHECK(even->averageSeconds == 4.0);
  CHECK(even->minimumExecution.executionIndex == 1u);
  CHECK(even->minimumExecution.candidateIndex == 1u);
  CHECK(even->minimumExecution.configuration ==
        alpakaTune::ParameterConfiguration{0.5f});
}
