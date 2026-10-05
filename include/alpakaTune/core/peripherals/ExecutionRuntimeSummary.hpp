// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0
#pragma once

#include "alpakaTune/core/TunerInfo.hpp"

#include <algorithm>
#include <numeric>
#include <span>
#include <vector>

namespace alpakaTune::detail {

[[nodiscard]] inline auto
summarizeExecutionRuntimes(std::span<ExecutedConfiguration const> executions)
    -> std::optional<ExecutionRuntimeSummary> {
  auto samples = std::vector<double>{};
  ExecutedConfiguration const *minimumExecution = nullptr;
  for (auto const &execution : executions) {
    if (!execution.runtimeSeconds)
      continue;
    samples.push_back(*execution.runtimeSeconds);
    if (!minimumExecution ||
        *execution.runtimeSeconds < *minimumExecution->runtimeSeconds)
      minimumExecution = &execution;
  }
  if (samples.empty())
    return std::nullopt;
  std::ranges::sort(samples);
  auto const middle = samples.size() / 2u;
  return ExecutionRuntimeSummary{
      .sampleCount = samples.size(),
      .minimumSeconds = samples.front(),
      .maximumSeconds = samples.back(),
      .medianSeconds = samples.size() % 2u ? samples[middle]
                                           : std::midpoint(samples[middle - 1u],
                                                           samples[middle]),
      .averageSeconds = std::accumulate(samples.begin(), samples.end(), 0.0) /
                        static_cast<double>(samples.size()),
      .minimumExecution = *minimumExecution};
}

} // namespace alpakaTune::detail
