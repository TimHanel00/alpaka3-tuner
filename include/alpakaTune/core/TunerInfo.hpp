// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "alpakaTune/interfaces/Strategy.hpp"

#include <cstddef>
#include <optional>
#include <string>

namespace alpakaTune {

enum class TunerCompletionReason {
  none,
  allConfigurations,
  maximumExecutions,
  maximumRetiredConfigurations,
};

[[nodiscard]] constexpr auto
completionReasonName(TunerCompletionReason reason) noexcept -> char const * {
  switch (reason) {
  case TunerCompletionReason::none:
    return "none";
  case TunerCompletionReason::allConfigurations:
    return "all_configurations";
  case TunerCompletionReason::maximumExecutions:
    return "maximum_executions";
  case TunerCompletionReason::maximumRetiredConfigurations:
    return "maximum_retired_configurations";
  }
  return "none";
}

struct TunerInfo {
  std::size_t candidateCount{};
  std::size_t rejectedCandidateCount{};
  std::size_t scheduledCandidateCount{};
  std::size_t measuredCandidateCount{};
  std::size_t retiredConfigurationCount{};
  std::size_t executionCount{};
  bool tuningComplete{};
  bool loadedFromCache{};
  bool executionBudgetReached{};
  TunerCompletionReason completionReason{TunerCompletionReason::none};
  std::optional<std::size_t> bestCandidateIndex;
  std::optional<ParameterConfiguration> bestConfiguration;
  std::optional<std::string> learnedStatus;
  std::size_t learnedAdapterUpdateCount{};
};

struct LaunchObservation {
  std::size_t candidateIndex{};
  ParameterConfiguration configuration;
  std::optional<double> runtimeSeconds;
  double recommendationSeconds{};
  bool measured{};
  bool tuningComplete{};
  bool loadedFromCache{};
  std::optional<std::string> learnedStatus;
  std::size_t learnedAdapterUpdateCount{};
};

} // namespace alpakaTune
