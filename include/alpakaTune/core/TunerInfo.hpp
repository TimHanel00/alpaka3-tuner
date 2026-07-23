// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "alpakaTune/core/TunerConfig.hpp"

#include <cstddef>
#include <optional>
#include <string>

namespace alpakaTune {

/** @brief Terminal reason recorded by fixed and offline tuner policies. */
enum class TunerCompletionReason {
  none,                         ///< No terminal state has been entered.
  offlineReplay,                ///< Offline mode loaded a persisted winner.
  allConfigurations,            ///< Every legal candidate was retired.
  maximumExecutions,            ///< Fixed-mode launch guard was reached.
  maximumRetiredConfigurations, ///< Fixed-mode retirement guard was reached.
};

/** @brief Return the stable persistence spelling of a completion reason. */
[[nodiscard]] constexpr auto
completionReasonName(TunerCompletionReason reason) noexcept -> char const * {
  switch (reason) {
  case TunerCompletionReason::none:
    return "none";
  case TunerCompletionReason::offlineReplay:
    return "offline_replay";
  case TunerCompletionReason::allConfigurations:
    return "all_configurations";
  case TunerCompletionReason::maximumExecutions:
    return "maximum_executions";
  case TunerCompletionReason::maximumRetiredConfigurations:
    return "maximum_retired_configurations";
  }
  return "none";
}

/** @brief Read-only snapshot of tuner policy, coverage, and diagnostics. */
struct TunerInfo {
  /** Execution mode used by this tuner. */
  TuningMode mode{TuningMode::onlineAdaptive};
  /** Cartesian candidate count before restrictions are evaluated lazily. */
  std::size_t candidateCount{};
  /** Candidates permanently rejected by tuning-space restrictions. */
  std::size_t rejectedCandidateCount{};
  /** Candidates currently owned by the active scheduler or fixed history. */
  std::size_t scheduledCandidateCount{};
  /** Non-rejected candidates with at least one retained timing sample. */
  std::size_t measuredCandidateCount{};
  /** Completed fixed records or completed adaptive activation visits. */
  std::size_t retiredConfigurationCount{};
  /** Total kernel launches, including warm-ups and production replays. */
  std::size_t executionCount{};
  /** Configured schedule horizon or fixed completion guard, if present. */
  std::optional<std::size_t> maximumExecutions;
  /** True only when the tuner entered an actual terminal state. */
  bool tuningComplete{};
  /** Whether compatible persistent state initialized this tuner. */
  bool loadedFromCache{};
  /** Whether fixed mode terminated specifically at the execution guard. */
  bool executionBudgetReached{};
  /** Current statistically best measured candidate, when one exists. */
  std::optional<std::size_t> bestCandidateIndex;
  /** Normalized parameters corresponding to bestCandidateIndex. */
  std::optional<ParameterConfiguration> bestConfiguration;
  /** Learned-model activation or explicit fallback status, when applicable. */
  std::optional<std::string> learnedStatus;
  /** Number of residual-adapter fits completed by learned hybrid. */
  std::size_t learnedAdapterUpdateCount{};
  /** First-time candidates accepted by shared admission policy. */
  std::size_t unseenAcceptedCount{};
  /** Previously measured candidates readmitted in adaptive mode. */
  std::size_t revisitAcceptedCount{};
  /** Proposals rejected because the candidate was already queue-resident. */
  std::size_t activeDuplicateRejectedCount{};
  /** Proposals rejected by a tuning-space restriction. */
  std::size_t restrictionRejectedCount{};
  /** Revisit proposals rejected by the adaptive sigmoid gate. */
  std::size_t revisitRejectedCount{};
  /** Revisit proposals rejected by the relative-score gate. */
  std::size_t scoreRejectedCount{};
};

/** @brief Per-call result returned by Tuner::enqueueObserved(). */
struct LaunchObservation {
  /** Exact Cartesian candidate launched by this call. */
  std::size_t candidateIndex{};
  /** Normalized parameter vector mapped to candidateIndex. */
  ParameterConfiguration configuration;
  /** Synchronized launch duration when timing was enabled. */
  std::optional<double> runtimeSeconds;
  /** Wall time spent obtaining and admitting this call's recommendation. */
  double recommendationSeconds{};
  /** Whether this call synchronized and produced runtimeSeconds. */
  bool measured{};
  /** Actual terminal-state snapshot after the launch. */
  bool tuningComplete{};
  /** Whether this tuner was restored from compatible persistence. */
  bool loadedFromCache{};
  /** Learned-model activation or fallback status, when applicable. */
  std::optional<std::string> learnedStatus;
  /** Residual-adapter fit count after the launch. */
  std::size_t learnedAdapterUpdateCount{};
};

} // namespace alpakaTune
