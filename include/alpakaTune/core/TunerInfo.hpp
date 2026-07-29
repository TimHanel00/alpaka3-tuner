// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "alpakaTune/core/TunerConfig.hpp"

#include <cstddef>
#include <optional>
#include <string>

namespace alpakaTune {

/** @brief Clock source used for synchronized kernel runtime observations. */
enum class RuntimeMeasurementSource {
  hostClock,   ///< Host wall clock around a synchronized Alpaka launch.
  deviceEvent, ///< Backend device-event timestamps around the kernel.
};

/** @brief Stable diagnostic spelling of a runtime measurement source. */
[[nodiscard]] constexpr auto
runtimeMeasurementSourceName(RuntimeMeasurementSource source) noexcept
    -> char const * {
  switch (source) {
  case RuntimeMeasurementSource::hostClock:
    return "host_clock";
  case RuntimeMeasurementSource::deviceEvent:
    return "device_event";
  }
  return "host_clock";
}

/** @brief Terminal reason recorded by tuner policies. */
enum class TunerCompletionReason {
  none,                         ///< No terminal state has been entered.
  offlineReplay,                ///< Offline mode loaded a persisted winner.
  allConfigurations,            ///< Every legal candidate was retired.
  maximumExecutions,            ///< Fixed-mode launch guard was reached.
  maximumRetiredConfigurations, ///< Fixed-mode retirement guard was reached.
  /** Fixed-mode admission could not accept repeated strategy proposals. */
  maximumConsecutiveStrategyRetries,
};

/** @brief Diagnostic emitted when timing overhead may dominate a short kernel.
 */
struct InstrumentationOverheadWarning {
  /** First synchronized kernel runtime which crossed the warning threshold. */
  double observedRuntimeSeconds{};
  /** Runtime below which the tuner emits this diagnostic. */
  double thresholdSeconds{200.0e-6};
  /** Representative lower instrumentation-overhead estimate. */
  double estimatedOverheadMinimumSeconds{20.0e-6};
  /** Representative upper instrumentation-overhead estimate. */
  double estimatedOverheadMaximumSeconds{40.0e-6};
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
  case TunerCompletionReason::maximumConsecutiveStrategyRetries:
    return "maximum_consecutive_strategy_retries";
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
  /** Records or adaptive visits retired in the current online run. */
  std::size_t retiredConfigurationCount{};
  /** Current-run kernel launches, including warm-ups and production replays. */
  std::size_t executionCount{};
  /** Launches counted toward this process run's adaptive horizon. */
  std::size_t adaptiveHorizonExecutionCount{};
  /** History-aware progress, or zero when no adaptive horizon is configured. */
  double adaptiveHorizonProgress{};
  /** Configured online-adaptive new-run horizon, when active. */
  std::optional<std::size_t> horizon;
  /** Configured online-fixed execution guard, if present. */
  std::optional<std::size_t> maximumExecutions;
  /** Rejected proposals allowed in one bounded refill attempt. */
  std::size_t maximumConsecutiveStrategyRetries{};
  /** Current number of strategy proposals rejected since the last admission. */
  std::size_t consecutiveStrategyRetries{};
  /** Bounded refill attempts which reached the configured retry limit. */
  std::size_t strategyRetryLimitReachedCount{};
  /** Measured adaptive fallbacks after bounded refill attempts were exhausted.
   */
  std::size_t adaptiveRetryFallbackCount{};
  /** Policy completion; horizon-less adaptive mode never signals completion. */
  bool tuningComplete{};
  /** Terminal reason when one exists; absent for an adaptive horizon. */
  std::optional<TunerCompletionReason> completionReason;
  /** Whether compatible persistent state initialized this tuner. */
  bool loadedFromCache{};
  /** Whether fixed mode terminated specifically at the execution guard. */
  bool executionBudgetReached{};
  /** Present after the first measured runtime below 200 microseconds. */
  std::optional<InstrumentationOverheadWarning> instrumentationOverheadWarning;
  /** Backend clock used for measured kernel runtimes. */
  RuntimeMeasurementSource runtimeMeasurementSource{
      RuntimeMeasurementSource::hostClock};
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
  /** Accepted proposals already represented by a queue-resident candidate. */
  std::size_t activeDuplicateAcceptedCount{};
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
  /** Backend clock used to produce runtimeSeconds. */
  RuntimeMeasurementSource runtimeMeasurementSource{
      RuntimeMeasurementSource::hostClock};
  /** Wall time spent obtaining and admitting this call's recommendation. */
  double recommendationSeconds{};
  /** Whether this call synchronized and produced runtimeSeconds. */
  bool measured{};
  /** Policy-completion snapshot; adaptive mode still measures afterward. */
  bool tuningComplete{};
  /** Whether this tuner was restored from compatible persistence. */
  bool loadedFromCache{};
  /** Learned-model activation or fallback status, when applicable. */
  std::optional<std::string> learnedStatus;
  /** Residual-adapter fit count after the launch. */
  std::size_t learnedAdapterUpdateCount{};
};

} // namespace alpakaTune
