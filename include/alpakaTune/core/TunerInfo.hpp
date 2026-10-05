// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "alpakaTune/space/SpaceConfig.hpp"

#include "alpakaTune/core/TunerConfig.hpp"
#include "alpakaTune/core/TuningMetric.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace alpakaTune {

template <typename TunablesType, typename Device,
          typename MetricPolicy = metric::Timing>
class Tuner;

namespace detail {
struct ConfigurationValidityState {
  bool valid{true};
  bool invalidationConsumed{};
};
} // namespace detail

/**
 * @brief Mutable application-owned validity flag for one tuner candidate.
 *
 * Copies share state, so every execution-history entry for the same candidate
 * changes together. An invalidation may be undone until the tuner consumes it
 * at the next enqueue. Once consumed, the candidate remains invalid for the
 * lifetime of the tuner.
 */
class ConfigurationValidity {
public:
  ConfigurationValidity()
      : m_state(std::make_shared<detail::ConfigurationValidityState>()) {}

  /** @brief Read the application's current validity decision. */
  [[nodiscard]] explicit operator bool() const noexcept {
    return m_state->valid;
  }

  /** @brief Change the decision before it is consumed by the tuner.
   * @throws std::logic_error when attempting to restore a consumed
   * invalidation.
   */
  auto operator=(bool valid) -> ConfigurationValidity & {
    if (valid && m_state->invalidationConsumed)
      throw std::logic_error{
          "A consumed configuration invalidation cannot be restored."};
    m_state->valid = valid;
    return *this;
  }

private:
  template <typename TunablesType, typename Device, typename MetricPolicy>
  friend class Tuner;

  explicit ConfigurationValidity(
      std::shared_ptr<detail::ConfigurationValidityState> state)
      : m_state(std::move(state)) {}

  std::shared_ptr<detail::ConfigurationValidityState> m_state;
};

/** @brief One successfully submitted tuner launch in process execution order.
 */
struct ExecutedConfiguration {
  /** Zero-based position in Tuner::history(). */
  std::size_t executionIndex{};
  /** Exact Cartesian candidate selected for the launch. */
  std::size_t candidateIndex{};
  /** Normalized parameter vector mapped to candidateIndex. */
  ParameterConfiguration configuration;
  /** Synchronized launch duration when timing was enabled. */
  std::optional<double> runtimeSeconds;
  /** Tuning objective attached to this launch, when one was provided. */
  std::optional<double> metricValue;
  /** Whether this launch supplied a sample used by tuning. */
  bool measured{};
  /** Application decision, shared by all entries for this candidate. */
  mutable ConfigurationValidity valid;
};

/** @brief Clock source used for synchronized kernel runtime observations. */
enum class RuntimeMeasurementSource {
  hostClock,   ///< Host wall clock around a synchronized Alpaka launch.
  deviceEvent, ///< Backend device-event timestamps around the kernel.
};

/** @brief Compile-time-selected source of tuner objective values. */
enum class TuningMetricKind {
  timing, ///< Built-in synchronized kernel-runtime instrumentation.
  custom, ///< Values supplied by the application through provideMetric().
};

/** @brief Stable diagnostic spelling of a tuning metric kind. */
[[nodiscard]] constexpr auto
tuningMetricKindName(TuningMetricKind kind) noexcept -> char const * {
  switch (kind) {
  case TuningMetricKind::timing:
    return "timing";
  case TuningMetricKind::custom:
    return "custom";
  }
  return "timing";
}

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

/** @brief Reason exploration ended. */
enum class TunerCompletionReason {
  none,              ///< No terminal state has been entered.
  offlineReplay,     ///< Offline exploration loaded measured history.
  allConfigurations, ///< Every legal candidate was retired.
  maximumExecutions, ///< Online exploration launch guard was reached.
  maximumRetiredConfigurations, ///< Online exploration retirement guard was
                                ///< reached.
  /** Fixed-mode admission could not accept repeated strategy proposals. */
  maximumConsecutiveStrategyRetries,
  /** Every measured candidate was invalidated or rejected. */
  candidateBudget,   ///< Generated catalog reached its size limit.
  plateau,           ///< Generation stopped after sustained lack of gain.
  generationStalled, ///< Bounded generation attempts found no new candidate.
  noValidConfiguration,
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
  case TunerCompletionReason::candidateBudget:
    return "candidate_budget";
  case TunerCompletionReason::plateau:
    return "plateau";
  case TunerCompletionReason::generationStalled:
    return "generation_stalled";
  case TunerCompletionReason::noValidConfiguration:
    return "no_valid_configuration";
  }
  return "none";
}

/** @brief Read-only snapshot of tuner policy, coverage, and diagnostics. */
struct TunerInfo {
  SpaceInfo space;
  /** Permission to search for new configurations. */
  ExplorationPolicy exploration{ExplorationPolicy::online};
  /** Whether configuration selection continues adapting. */
  SelectionPolicy selection{SelectionPolicy::adaptive};
  /** Whether exploration has ended, independently of continuing selection. */
  bool explorationComplete{};
  /** Whether a valid winner is locked for replay. */
  bool selectionLocked{};
  /** Cartesian candidate count before restrictions are evaluated lazily. */
  std::size_t candidateCount{};
  /** Candidates permanently rejected by tuning-space restrictions. */
  std::size_t rejectedCandidateCount{};
  /** Executed candidates permanently invalidated by the application. */
  std::size_t userInvalidatedCandidateCount{};
  /** Candidates rejected because their last launch received no metric. */
  std::size_t missingMetricCandidateCount{};
  /** Candidates currently owned by the active scheduler or fixed history. */
  std::size_t scheduledCandidateCount{};
  /** Non-rejected candidates with at least one retained metric sample. */
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
  /** Configured online exploration execution guard, if present. */
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
  /** Exploration completion; adaptive selection may continue afterward. */
  bool tuningComplete{};
  /** Exploration completion reason when one exists. */
  std::optional<TunerCompletionReason> completionReason;
  /** Whether compatible persistent state initialized this tuner. */
  bool loadedFromCache{};
  /** Whether exploration ended specifically at the execution guard. */
  bool executionBudgetReached{};
  /** Present after the first measured runtime below 200 microseconds. */
  std::optional<InstrumentationOverheadWarning> instrumentationOverheadWarning;
  /** Compile-time-selected source of the minimized objective. */
  TuningMetricKind metricKind{TuningMetricKind::timing};
  /** Runtime metric label stored in persistent histories. */
  std::string metricName{"runtime_seconds"};
  /** Backend clock used for runtime metrics; absent for custom metrics. */
  std::optional<RuntimeMeasurementSource> runtimeMeasurementSource;
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
  /** Proposals rejected because the application invalidated the candidate. */
  std::size_t userInvalidatedRejectedCount{};
  /** Proposals rejected because a required custom metric was not supplied. */
  std::size_t missingMetricRejectedCount{};
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
  /** Objective available when this call returned; custom metrics arrive later.
   */
  std::optional<double> metricValue;
  /** Backend clock used to produce runtimeSeconds, when timing is active. */
  std::optional<RuntimeMeasurementSource> runtimeMeasurementSource;
  /** Wall time spent obtaining and admitting this call's recommendation. */
  double recommendationSeconds{};
  /** Whether this call already produced a tuning sample when it returned. */
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
