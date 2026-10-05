// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "alpakaTune/adjust/DerivedThreadSpec.hpp"
#include "alpakaTune/core/Persistence.hpp"
#include "alpakaTune/core/TunerConfig.hpp"
#include "alpakaTune/core/TunerInfo.hpp"
#include "alpakaTune/core/peripherals/BenefitBudget.hpp"
#include "alpakaTune/core/peripherals/CandidateQueue.hpp"
#include "alpakaTune/core/peripherals/ExecutionRuntimeSummary.hpp"
#include "alpakaTune/core/timing/KernelTimer.hpp"
#include "alpakaTune/model/LearnedModelContext.hpp"
#include "alpakaTune/space/CandidateSpace.hpp"
#include "alpakaTune/store/RuntimeHistory.hpp"
#include "alpakaTune/strategy/StrategyFactory.hpp"
#include "alpakaTune/tunable/Tunables.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#if ALPAKA_TUNE_HAS_JSON
#include <nlohmann/json.hpp>
#endif

namespace alpakaTune {

namespace detail {

inline constexpr FixedString frameExtentName{"frameExtent"};
inline constexpr FixedString numFramesName{"numFrames"};
inline constexpr FixedString numThreadsName{"numThreads"};
inline constexpr FixedString numBlocksName{"numBlocks"};

struct BestImprovement {
  std::size_t candidateIndex{};
  std::size_t executionCount{};
  std::size_t retiredConfigurationCount{};
  double metricValue{};
  double elapsedSeconds{};
};

template <typename... Entries>
inline constexpr std::size_t customMetricTokenCount =
    (std::size_t{0u} + ... +
     (isCustomMetricToken<std::remove_cvref_t<Entries>> ? 1u : 0u));

template <typename... Entries> struct MetricPolicyFor {
  using type = metric::Timing;
};
template <typename First, typename... Rest>
struct MetricPolicyFor<First, Rest...> {
  using type =
      std::conditional_t<isCustomMetricToken<std::remove_cvref_t<First>>,
                         std::remove_cvref_t<First>,
                         typename MetricPolicyFor<Rest...>::type>;
};
template <typename... Entries>
using SelectedMetricPolicy = typename MetricPolicyFor<Entries...>::type;

template <typename Device, bool Enabled> struct MeasurementTimerStorage {};

template <typename Device> struct MeasurementTimerStorage<Device, true> {
  std::optional<timing::KernelTimer<Device>> timer;
};

/** @brief Normalized sigmoid used to admit adaptive revisits.
 *
 * @param progress Execution progress; clamped to [0, 1].
 * @param steepness Positive sigmoid steepness.
 * @return A probability normalized to exactly 0 at progress 0 and exactly 1
 * at progress 1.
 */
[[nodiscard]] inline auto normalizedLogisticAdmission(double progress,
                                                      double steepness)
    -> double {
  progress = std::clamp(progress, 0.0, 1.0);
  auto const sigmoid = [](double value) {
    return 1.0 / (1.0 + std::exp(-value));
  };
  auto const lower = sigmoid(-steepness / 2.0);
  auto const upper = sigmoid(steepness / 2.0);
  return std::clamp((sigmoid(steepness * (progress - 0.5)) - lower) /
                        (upper - lower),
                    0.0, 1.0);
}

/** @brief Exponentially interpolate the adaptive score temperature.
 * @param progress Execution progress; clamped to [0, 1].
 * @param start Temperature at progress zero.
 * @param end Temperature at and after the schedule horizon.
 */
[[nodiscard]] inline auto adaptiveScoreTemperature(double progress,
                                                   double start, double end)
    -> double {
  progress = std::clamp(progress, 0.0, 1.0);
  return start * std::pow(end / start, progress);
}

/** @brief Stretch one adaptive horizon over a history-aware progress range.
 *
 * A fresh tuner keeps the original [0, 1] schedule. A tuner initialized from
 * compatible measured history maps its new-run progress onto [offset, 1].
 */
[[nodiscard]] inline auto adaptiveProgressWithActiveHistory(double progress,
                                                            bool activeHistory,
                                                            double offset)
    -> double {
  progress = std::clamp(progress, 0.0, 1.0);
  if (!activeHistory)
    return progress;
  return offset + (1.0 - offset) * progress;
}

/** @brief Boltzmann admission probability relative to the current best.
 *
 * The best candidate and any equally fast candidate receive probability one.
 * Slower candidates decay according to their relative slowdown, keeping the
 * gate invariant under uniform runtime scaling.
 */
[[nodiscard]] inline auto relativeScoreAdmission(double score, double best,
                                                 double temperature) -> double {
  if (!std::isfinite(score) || !std::isfinite(best) || score < 0.0 ||
      best < 0.0 || !std::isfinite(temperature) || temperature <= 0.0)
    throw std::invalid_argument{
        "Adaptive admission received an invalid score."};
  auto const scale = std::max(best, std::numeric_limits<double>::min());
  auto const relativeSlowdown = std::max(0.0, score / scale - 1.0);
  return std::exp(-relativeSlowdown / temperature);
}

template <FixedString Name>
inline constexpr bool isReservedLaunchName =
    sameName<Name, frameExtentName> || sameName<Name, numFramesName> ||
    sameName<Name, numThreadsName> || sameName<Name, numBlocksName>;

template <typename T> [[nodiscard]] auto typeName() -> std::string {
#if defined(__clang__) || defined(__GNUC__)
  std::string_view signature{__PRETTY_FUNCTION__};
  auto const begin = signature.find("T = ") + 4u;
  auto const end = signature.find_first_of(";]", begin);
  return std::string{signature.substr(begin, end - begin)};
#elif defined(_MSC_VER)
  std::string_view signature{__FUNCSIG__};
  auto const begin = signature.find("typeName<") + 9u;
  auto const end = signature.find(">", begin);
  return std::string{signature.substr(begin, end - begin)};
#else
  return typeid(T).name();
#endif
}

template <typename T>
[[nodiscard]] auto printable(T const &value) -> std::string {
  if constexpr (std::convertible_to<T, std::string_view>)
    return std::string{std::string_view{value}};
  else if constexpr (requires { value.toString(); })
    return value.toString();
  else if constexpr (requires { value.getName(); })
    return value.getName();
  else if constexpr (requires { std::to_string(value); })
    return std::to_string(value);
  else
    return typeName<T>();
}

template <typename T>
[[nodiscard]] auto identityName(T const &value) -> std::string {
  if constexpr (std::convertible_to<T, std::string_view>)
    return std::string{std::string_view{value}};
  else if constexpr (requires { value.getName(); })
    return std::string{value.getName()};
  else
    return printable(value);
}

inline auto hashFingerprint(std::string_view text) -> std::string {
  std::uint64_t value = 14695981039346656037ull;
  for (auto const character : text) {
    value ^= static_cast<unsigned char>(character);
    value *= 1099511628211ull;
  }
  std::ostringstream output;
  output << std::hex << std::setw(16) << std::setfill('0') << value;
  return output.str();
}

inline auto fileFingerprint(std::filesystem::path const &path) -> std::string {
  auto input = std::ifstream{path, std::ios::binary};
  if (!input)
    return "unavailable";
  auto value = std::uint64_t{14695981039346656037ull};
  auto buffer = std::array<char, 64u * 1024u>{};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    auto const count = input.gcount();
    for (std::streamsize index = 0; index < count; ++index) {
      value ^=
          static_cast<unsigned char>(buffer[static_cast<std::size_t>(index)]);
      value *= 1099511628211ull;
    }
  }
  std::ostringstream output;
  output << std::hex << std::setw(16) << std::setfill('0') << value;
  return output.str();
}

/** Mix a stable context fingerprint into a configured base seed. */
[[nodiscard]] inline auto contextSeed(std::uint64_t baseSeed,
                                      std::string_view fingerprint) noexcept
    -> std::uint64_t {
  auto seed = std::uint64_t{14695981039346656037ull} ^ baseSeed;
  for (auto const character : fingerprint) {
    seed ^= static_cast<unsigned char>(character);
    seed *= 1099511628211ull;
  }
  return seed;
}

/** Create the explicitly requested nondeterministic process-local base seed. */
[[nodiscard]] inline auto nondeterministicSeed() -> std::uint64_t {
  auto source = std::random_device{};
  auto seed = std::uint64_t{source()};
  seed = (seed << 32u) ^ std::uint64_t{source()};
  return seed;
}

template <FixedString Name, typename Value> struct SelectedCompileValue {
  static constexpr auto name = Name;
  using value_type = Value;
};

template <typename T> struct LaunchValue {
  [[nodiscard]] static constexpr auto get() -> T { return {}; }
};

template <typename T, T... Values>
struct LaunchValue<std::integer_sequence<T, Values...>> {
  [[nodiscard]] static constexpr auto get() -> alpaka::CVec<T, Values...> {
    return {};
  }
};

template <typename T> [[nodiscard]] constexpr auto launchValue() {
  return LaunchValue<T>::get();
}

template <FixedString Name, typename... Values>
consteval auto findSelectedCompileIndex() -> std::size_t {
  static_assert((sameName<Name, Values::name> || ...),
                "The compile-time marker was not selected.");
  constexpr auto matches = std::array{sameName<Name, Values::name>...};
  for (auto index = std::size_t{}; index < matches.size(); ++index)
    if (matches[index])
      return index;
  return matches.size();
}

template <FixedString Name, typename... Values> struct FindSelectedCompile {
  using selected_type =
      std::tuple_element_t<findSelectedCompileIndex<Name, Values...>(),
                           std::tuple<Values...>>;
  using type = typename selected_type::value_type;
};

template <typename Values, std::size_t Index = 0u, typename Callable>
void dispatchCVal(std::size_t selectedIndex, Callable &&callable) {
  if constexpr (Index == Values::size) {
    throw std::out_of_range{
        "A compile-time tunable candidate index is out of range."};
  } else {
    if (selectedIndex == Index) {
      using Selected = std::tuple_element_t<Index, typename Values::values>;
      std::invoke(std::forward<Callable>(callable),
                  std::type_identity<Selected>{});
      return;
    }
    dispatchCVal<Values, Index + 1u>(selectedIndex,
                                     std::forward<Callable>(callable));
  }
}

template <typename Tuple, std::size_t Index = 0u, typename Callable>
void forEachTupleType(Callable &&callable) {
  if constexpr (Index < std::tuple_size_v<Tuple>) {
    std::invoke(std::forward<Callable>(callable),
                std::type_identity<std::tuple_element_t<Index, Tuple>>{});
    forEachTupleType<Tuple, Index + 1u>(std::forward<Callable>(callable));
  }
}

} // namespace detail

/** @brief Device-bound owner of scheduling, measurement, and persistence.
 *
 * Each enqueue call performs exactly one application-requested kernel launch.
 * The tuner selects the candidate and may time it, but never decides how often
 * the surrounding application calls enqueue. One instance binds lazily to one
 * launch prototype and one persistent fingerprint.
 */
template <typename TunablesType, typename Device, typename MetricPolicy>
class Tuner {
public:
  using Traits = detail::TunablesTraits<TunablesType>;
  using Entries = typename Traits::entries_type;
  static constexpr std::size_t dimensionCount = Traits::dimensionCount;
  // The executor is part of the launch identity. Histories produced before
  // that identity was persisted must not be reused by a newer tuner.
  static constexpr int completeHistorySchemaVersion = 13;
  static constexpr bool usesCustomMetric =
      detail::isCustomMetricToken<MetricPolicy>;
  static constexpr bool tunesFrameSpec =
      Traits::template has<detail::numFramesName> ||
      Traits::template has<detail::frameExtentName>;
  static constexpr bool tunesThreadSpec =
      Traits::template has<detail::numBlocksName> ||
      Traits::template has<detail::numThreadsName>;
  static_assert(!(tunesFrameSpec && tunesThreadSpec),
                "A tuner cannot combine FrameSpec and ThreadSpec parameters.");
  using MeasurementTimer = detail::timing::KernelTimer<Device>;

  /** @brief Construct a tuner from a validated policy and identity components.
   *
   * Prefer makeTuner(); this constructor is public to preserve the aggregate
   * factory's concrete return type.
   */
  Tuner(TunerConfig defaults, std::shared_ptr<detail::HistoryStore> history,
        std::shared_ptr<detail::CompleteHistoryStore> completeHistory,
        TunablesType tunables, Device device,
        std::vector<std::string> identityEntries, std::string metricName,
        MetricPolicy metricPolicy = {})
      : m_defaults(std::move(defaults)), m_history(std::move(history)),
        m_completeHistory(std::move(completeHistory)),
        m_tunables(std::move(tunables)), m_device(std::move(device)),
        m_identityEntries(std::move(identityEntries)),
        m_metricName(std::move(metricName)),
        m_metricPolicy(std::move(metricPolicy)),
        m_baseRandomSeed(m_defaults.randomSeed
                             ? *m_defaults.randomSeed
                             : detail::nondeterministicSeed()),
        m_random(0u) {
    if (m_metricName.empty())
      throw std::invalid_argument{"A tuning metric needs a name."};
    if (m_defaults.budget) {
      if constexpr (!detail::isElapsedTimeMetric<MetricPolicy>)
        throw std::invalid_argument{
            "Benefit budgeting requires timing or elapsedTimeMetric()."};
      m_budget.emplace(*m_defaults.budget);
    }
    initialiseDimensions();
  }

  /** @brief Revise future work without reopening completed exploration. */
  void setRemainingLaunches(std::uint64_t launches) {
    if (!m_budget)
      throw std::logic_error{"This tuner has no application-runtime budget."};
    m_budget->remaining = launches;
    m_budgetNextCheck = m_budget->launches;
  }

  /** @brief Runtime label of the compile-time-selected minimization metric. */
  [[nodiscard]] auto metricName() const noexcept -> std::string_view {
    return m_metricName;
  }
  /** @brief Compile-time-selected source of objective measurements. */
  [[nodiscard]] static constexpr auto metricKind() noexcept
      -> TuningMetricKind {
    if constexpr (usesCustomMetric)
      return TuningMetricKind::custom;
    else
      return TuningMetricKind::timing;
  }
  /** @brief Timing clock, absent when the application owns measurement. */
  [[nodiscard]] static constexpr auto runtimeMeasurementSource() noexcept
      -> std::optional<RuntimeMeasurementSource> {
    if constexpr (usesCustomMetric)
      return std::nullopt;
    else
      return MeasurementTimer::measurementSource();
  }

  /** Whether exploration ended. Adaptive selection continues afterward. */
  [[nodiscard]] auto isTuningComplete() const noexcept -> bool {
    return m_explorationComplete;
  }
  /** Backward-compatible policy-completion query.
   *
   * Equivalent to isTuningComplete(). It never controls application lifetime.
   */
  [[nodiscard]] auto completed() const noexcept -> bool {
    return isTuningComplete();
  }
  /** @brief Explain why exploration ended.
   * @throws std::logic_error while exploration remains active.
   */
  [[nodiscard]] auto completionReason() const -> TunerCompletionReason {
    if (!m_explorationComplete)
      throw std::logic_error{"Exploration has not completed."};
    return m_completionReason;
  }
  /** @brief Whether compatible persistence initialized this tuner. */
  [[nodiscard]] auto loadedFromCache() const noexcept -> bool {
    return m_loadedFromCache;
  }
  /** @brief Configured proposal strategy, including an explicit learned kind.
   */
  [[nodiscard]] auto strategyKind() const noexcept -> StrategyKind {
    return m_defaults.strategy;
  }
  /** @brief Return the current measured winner's Cartesian index.
   * @throws std::logic_error before exploration completion or without a valid
   * winner.
   */
  [[nodiscard]] auto bestCandidateIndex() const -> std::size_t {
    if (!m_explorationComplete)
      throw std::logic_error{"Exploration has not completed."};
    if (auto const best = currentBestCandidate())
      return *best;
    throw std::logic_error{"The tuner has no measured winner."};
  }
  /** @brief Return the current measured winner's normalized parameter vector.
   */
  [[nodiscard]] auto bestConfiguration() const -> ParameterConfiguration {
    return candidateConfiguration(bestCandidateIndex());
  }
  /** @brief Map an exact Cartesian candidate to normalized parameters.
   * @throws std::out_of_range if @p candidate is outside this tuner.
   */
  [[nodiscard]] auto candidateConfiguration(std::size_t candidate) const
      -> ParameterConfiguration {
    if (candidate >= m_candidateCount)
      throw std::out_of_range{"The candidate index is outside this tuner."};
    return normalizedConfiguration(candidate);
  }
  /** @brief Ordered successfully submitted launches in this process.
   *
   * Record metadata is read-only; each record's validity flag remains mutable.
   */
  [[nodiscard]] auto history() const noexcept
      -> std::span<ExecutedConfiguration const> {
    return m_executionHistory;
  }
  /** @brief Summarize recorded launch runtimes and the minimum's configuration.
   *
   * Uses raw current-process execution history, excluding untimed launches and
   * restored samples. Returns nullopt when no launch runtime was recorded.
   * The minimum sample need not belong to the tuner's statistical winner.
   */
  [[nodiscard]] auto executionRuntimeSummary() const
      -> std::optional<ExecutionRuntimeSummary> {
    return detail::summarizeExecutionRuntimes(history());
  }
  /** @brief Most recent configuration that was successfully submitted.
   * @throws std::logic_error before the first successful enqueue.
   */
  [[nodiscard]] auto lastConfig() const -> ExecutedConfiguration const & {
    if (m_executionHistory.empty())
      throw std::logic_error{"The tuner has not launched a configuration."};
    return m_executionHistory.back();
  }
  /** @brief Cartesian index used by the most recent successful enqueue call. */
  [[nodiscard]] auto lastCandidateIndex() const noexcept -> std::size_t {
    return m_executionHistory.empty()
               ? std::numeric_limits<std::size_t>::max()
               : m_executionHistory.back().candidateIndex;
  }
  /** @brief Snapshot current coverage, policy, best, and admission counters. */
  [[nodiscard]] auto info() const -> TunerInfo {
    auto measured = std::size_t{};
    for (std::size_t candidate = 0u; candidate < m_histories.size();
         ++candidate)
      if (!m_rejected.empty() && !m_rejected.at(candidate) &&
          candidateUsable(candidate) && !m_histories.at(candidate).empty())
        ++measured;
    auto result = TunerInfo{
        .exploration = m_defaults.exploration,
        .selection = m_defaults.selection,
        .explorationComplete = m_explorationComplete,
        .selectionLocked = m_terminal && currentBestCandidate().has_value(),
        .candidateCount = m_candidateCount,
        .rejectedCandidateCount = m_rejectedCount,
        .userInvalidatedCandidateCount = m_userInvalidatedCount,
        .missingMetricCandidateCount = m_missingMetricCount,
        .scheduledCandidateCount = m_scheduledCount,
        .measuredCandidateCount = measured,
        .retiredConfigurationCount = m_retiredConfigurationCount,
        .executionCount = m_executionCount,
        .adaptiveHorizonExecutionCount = m_adaptiveHorizonExecutionCount,
        .adaptiveHorizonProgress =
            m_defaults.selection == SelectionPolicy::adaptive
                ? adaptiveProgress()
                : 0.0,
        .horizon = m_defaults.selection == SelectionPolicy::adaptive
                       ? m_defaults.horizon
                       : std::nullopt,
        .maximumExecutions = m_defaults.maximumExecutions,
        .maximumConsecutiveStrategyRetries =
            m_defaults.maximumConsecutiveStrategyRetries,
        .consecutiveStrategyRetries = m_consecutiveStrategyRetries,
        .strategyRetryLimitReachedCount = m_strategyRetryLimitReachedCount,
        .adaptiveRetryFallbackCount = m_adaptiveRetryFallbackCount,
        .tuningComplete = isTuningComplete(),
        .completionReason =
            m_explorationComplete
                ? std::optional<TunerCompletionReason>{m_completionReason}
                : std::nullopt,
        .loadedFromCache = m_loadedFromCache,
        .executionBudgetReached = m_executionBudgetReached,
        .instrumentationOverheadWarning = m_instrumentationOverheadWarning,
        .metricKind = metricKind(),
        .metricName = m_metricName,
        .runtimeMeasurementSource = runtimeMeasurementSource(),
        .bestCandidateIndex = std::nullopt,
        .bestConfiguration = std::nullopt,
        .learnedStatus = std::nullopt,
        .learnedAdapterUpdateCount = 0u,
        .unseenAcceptedCount = m_unseenAcceptedCount,
        .revisitAcceptedCount = m_revisitAcceptedCount,
        .activeDuplicateAcceptedCount = m_activeDuplicateAcceptedCount,
        .restrictionRejectedCount = m_restrictionRejectedCount,
        .userInvalidatedRejectedCount = m_userInvalidatedRejectedCount,
        .missingMetricRejectedCount = m_missingMetricRejectedCount,
        .revisitRejectedCount = m_revisitRejectedCount,
        .scoreRejectedCount = m_scoreRejectedCount};
    if (auto const best = currentBestCandidate()) {
      result.bestCandidateIndex = *best;
      result.bestConfiguration = normalizedConfiguration(*best);
    } else if (m_bestCandidate != std::numeric_limits<std::size_t>::max() &&
               candidateUsable(m_bestCandidate)) {
      result.bestCandidateIndex = m_bestCandidate;
      result.bestConfiguration = normalizedConfiguration(m_bestCandidate);
    }
    if (auto const *learned =
            dynamic_cast<LearnedHybridStrategy const *>(m_strategy.get())) {
      result.learnedStatus = learnedHybridStatusName(learned->status());
      result.learnedAdapterUpdateCount = learned->adapterUpdateCount();
    }
    if (m_budget)
      result.budget =
          BudgetInfo{m_budget->remaining,          m_budget->measurements,
                     m_budget->productionLaunches, m_budget->spent,
                     m_budget->ceiling(),          m_budget->expectedSavings,
                     m_budgetStoppingReason};
    result.space = m_space.info();
    return result;
  }
  /** @brief Current robust statistics for one Cartesian candidate. */
  [[nodiscard]] auto candidateRuntimeStatistics(std::size_t candidate) const
      -> detail::RuntimeStatistics {
    if (m_histories.empty())
      throw std::logic_error{"The tuner has not launched a candidate yet."};
    return m_histories.at(candidate).statistics();
  }
  /** @brief Retained rolling timing window for one Cartesian candidate. */
  [[nodiscard]] auto candidateRuntimeSamples(std::size_t candidate) const
      -> std::span<double const> {
    if (m_histories.empty())
      throw std::logic_error{"The tuner has not launched a candidate yet."};
    return m_histories.at(candidate).samples();
  }

  /** @brief Current robust statistics for one candidate's tuning metric. */
  [[nodiscard]] auto candidateMetricStatistics(std::size_t candidate) const
      -> detail::RuntimeStatistics {
    return candidateRuntimeStatistics(candidate);
  }
  /** @brief Retained rolling metric values for one Cartesian candidate. */
  [[nodiscard]] auto candidateMetricSamples(std::size_t candidate) const
      -> std::span<double const> {
    return candidateRuntimeSamples(candidate);
  }

  /** @brief Attach one application objective to the most recent enqueue.
   *
   * Lower values are better. The value must be finite and non-negative. This
   * member exists only for tuners constructed with customMetric().
   *
   * @throws std::logic_error if no most-recent launch is awaiting a metric or
   * if that launch already received one.
   */
  void provideMetric(double value)
    requires(usesCustomMetric)
  {
    requirePendingMetric();
    if (!std::isfinite(value) || value < 0.0)
      throw std::invalid_argument{
          "A custom tuning metric must be finite and non-negative."};
    auto const pending = *m_pendingMetric;
    auto const metricStart = m_budget && pending.measure
                                 ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};
    if (pending.measure)
      recordMetric(pending.candidateIndex, value, pending.beginActivation,
                   pending.endActivation);
    auto &execution = m_executionHistory.at(pending.executionIndex);
    execution.metricValue = value;
    execution.measured = pending.measure;
    m_pendingMetric.reset();
    if (m_budget && pending.measure) {
      auto const metricCost =
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        metricStart)
              .count();
      auto const cost =
          m_budgetPendingHostCost + metricCost +
          std::max(0.0, value - m_budgetPendingIncumbent) +
          m_budget->policy.measurementCostHintSeconds.value_or(0.0);
      m_budget->charge(cost);
      m_budget->observeCost(m_budgetPendingHostCost + metricCost);
    }
  }

  /** @brief Evaluate the configured scoring function for the latest launch.
   * The function is called only after submission eligibility has been checked.
   * Exceptions and invalid scores leave that launch awaiting a valid metric.
   */
  template <typename... Values>
    requires(detail::acceptsMetricInputs<MetricPolicy, Values...>)
  void provideMetrics(Values &&...values) {
    requirePendingMetric();
    if (m_budget && m_pendingMetric->measure) {
      auto const start = std::chrono::steady_clock::now();
      auto const value =
          std::invoke(m_metricPolicy.function, std::forward<Values>(values)...);
      m_budgetPendingHostCost += std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - start)
                                     .count();
      provideMetric(value);
    } else {
      provideMetric(std::invoke(m_metricPolicy.function,
                                std::forward<Values>(values)...));
    }
  }

  template <typename Queue, typename FrameSpec, typename Kernel,
            typename... Args>
  /** @brief Select and perform exactly one kernel launch.
   *
   * The application owns call count and lifetime. Depending on mode and queue
   * state, the launch may be a warm-up, measurement, adaptive revisit, or
   * terminal winner replay.
   *
   * @warning A measured GPU launch synchronizes the queue and commonly adds
   * approximately 20--40 microseconds of instrumentation overhead. After the
   * first measured runtime below 200 microseconds, info() exposes a persistent
   * instrumentationOverheadWarning for this tuner context and the tuner emits
   * the same warning once through std::clog.
   * @note CUDA and HIP report timing-enabled device-event durations. Host and
   * SYCL CPU use an explicitly reported synchronized host-clock fallback.
   * SYCL GPUs report command-profiling timestamps. Construct the queue passed
   * here with device.makeQueue(alpaka::queueKind::nonBlocking,
   * alpaka::timing::enabled).
   */
  void enqueue(Queue const &queue, FrameSpec const &frameSpec,
               alpaka::KernelBundle<Kernel, Args...> const &prototype) {
    if (m_budget) {
      static_cast<void>(enqueueBudgeted(queue, frameSpec, prototype, false));
      return;
    }
    if (m_defaults.replayFastPath &&
        (m_terminal || m_defaults.exploration == ExplorationPolicy::offline) &&
        tryFastReplay(queue, frameSpec, prototype))
      return;
    static_cast<void>(enqueueObserved(queue, frameSpec, prototype));
  }

  template <typename Queue, typename FrameSpec, typename Kernel,
            typename... Args>
  /** @brief Perform one launch and return its selection/timing diagnostics.
   *
   * Measured GPU launches have the synchronization overhead documented on
   * enqueue(); applications that need the once-only short-kernel diagnostic
   * can also inspect info().instrumentationOverheadWarning.
   */
  [[nodiscard]] auto
  enqueueObserved(Queue const &queue, FrameSpec const &frameSpec,
                  alpaka::KernelBundle<Kernel, Args...> const &prototype)
      -> LaunchObservation {
    if (m_budget)
      return enqueueBudgeted(queue, frameSpec, prototype);
    m_recommendationSecondsSinceLastLaunch = 0.0;
    if (!(queue.getDevice() == m_device))
      throw std::invalid_argument{
          "The queue device does not match this tuner's device."};
    validatePrototype(prototype);
    auto const fingerprint =
        makeFingerprint<FrameSpec, decltype(prototype)>(frameSpec, prototype);
    bindFingerprint(
        fingerprint,
        detail::typeName<
            typename std::remove_cvref_t<decltype(prototype)>::KernelFn>(),
        launchDescription(frameSpec));
    bindLaunchValidator<Queue>(frameSpec, prototype);
    initialiseScheduling();
    reconcileUserInvalidations();
    reconcileMissingMetric();
    if (m_terminal &&
        m_bestCandidate == std::numeric_limits<std::size_t>::max())
      throw std::logic_error{
          "The tuner terminated without a measured candidate to launch."};

    if constexpr (!usesCustomMetric &&
                  !std::same_as<ALPAKA_TYPEOF(queue.getTiming()),
                                alpaka::timing::Enabled>) {
      if (!m_defaults.replayFastPath || !m_terminal)
        throw std::invalid_argument{
            "A timing-disabled queue requires replayFastPath and a terminal "
            "offline or online-fixed winner."};
    }

    if (!m_explorationComplete && executionBudgetReached())
      finishTuningAtBudget(TunerCompletionReason::maximumExecutions);
    if (!m_explorationComplete && retiredConfigurationBudgetReached())
      finishTuningAtBudget(TunerCompletionReason::maximumRetiredConfigurations);

    using Bundle = std::remove_cvref_t<decltype(prototype)>;
    initialiseCompileVariants<Queue, FrameSpec, Bundle>();
    auto const selection =
        m_terminal ? detail::CandidateQueue::Selection{m_bestCandidate, false,
                                                       false, false}
                   : nextCandidate();
    auto const reuseLaunch = m_explorationComplete && !m_terminal;
    auto const candidate = selection.candidateIndex;
    LaunchCall<Queue, FrameSpec, Bundle> call{.tuner = this,
                                              .queue = &queue,
                                              .frameSpec = &frameSpec,
                                              .prototype = &prototype,
                                              .runtimeSeconds = std::nullopt};
    auto const key = compileVariantKey(indicesFor(candidate));
    m_compileVariants.at(m_compileVariantIndices.at(key))(
        &call, candidate, selection.measure, selection.beginActivation,
        selection.endActivation, !(m_defaults.replayFastPath && m_terminal));
    auto const executionIndex = recordExecution(candidate, call.runtimeSeconds);
    if (reuseLaunch) {
      ++m_reuseLaunchCount;
      if (m_reuseLaunchCount % m_defaults.adaptiveProbeInterval == 0u)
        m_nextProbeCandidate = (candidate + 1u) % m_candidateCount;
    }
    if constexpr (usesCustomMetric)
      m_pendingMetric =
          PendingMetric{.executionIndex = executionIndex,
                        .candidateIndex = candidate,
                        .measure = selection.measure,
                        .beginActivation = selection.beginActivation,
                        .endActivation = selection.endActivation};

    if (m_defaults.selection == SelectionPolicy::fixed && !m_terminal &&
        m_queue && m_queue->empty() && allFixedCandidatesAdmitted())
      finishTuning();

    auto const tunerInfo = info();
    return LaunchObservation{
        .candidateIndex = candidate,
        .configuration = normalizedConfiguration(candidate),
        .runtimeSeconds = call.runtimeSeconds,
        .metricValue = call.runtimeSeconds,
        .runtimeMeasurementSource = runtimeMeasurementSource(),
        .recommendationSeconds = m_recommendationSecondsSinceLastLaunch,
        .measured = call.runtimeSeconds.has_value(),
        .tuningComplete = isTuningComplete(),
        .loadedFromCache = m_loadedFromCache,
        .learnedStatus = tunerInfo.learnedStatus,
        .learnedAdapterUpdateCount = tunerInfo.learnedAdapterUpdateCount};
  }

  template <typename Queue, typename FrameSpec, typename Kernel,
            typename... Args>
  void tune(Queue const &queue, FrameSpec const &frameSpec,
            alpaka::KernelBundle<Kernel, Args...> const &prototype) {
    enqueue(queue, frameSpec, prototype);
  }

private:
  void ensureBudgetStorage() {
    auto const oldSize = m_budgetStatistics.size();
    m_budgetStatistics.resize(m_candidateCount);
    m_budgetInferior.resize(m_candidateCount, false);
    m_budgetEpochFresh.resize(m_candidateCount, true);
    m_budgetVisitSamples.resize(m_candidateCount, 0u);
    m_budgetWarmups.resize(m_candidateCount, 0u);
    for (auto candidate = oldSize; candidate < m_candidateCount; ++candidate)
      m_budgetStatistics[candidate] = m_histories[candidate].statistics();
  }

  void refreshBudgetIncumbent() {
    m_budgetIncumbent.reset();
    auto estimate = std::numeric_limits<double>::infinity();
    ensureBudgetStorage();
    for (std::size_t candidate = 0u; candidate < m_candidateCount; ++candidate)
      if (candidateUsable(candidate) && !m_rejected[candidate] &&
          !m_histories[candidate].empty() && m_budgetEpochFresh[candidate] &&
          m_budgetStatistics[candidate].median < estimate) {
        estimate = m_budgetStatistics[candidate].median;
        m_budgetIncumbent = candidate;
      }
    if (m_budgetIncumbent) {
      ensureBudgetStorage();
      m_budget->establishBaseline(
          m_budgetStatistics[*m_budgetIncumbent].median);
      m_bestCandidate = *m_budgetIncumbent;
    }
  }

  [[nodiscard]] auto budgetSampleLimit() const -> std::size_t {
    return m_defaults.selection == SelectionPolicy::fixed
               ? m_defaults.runsPerCandidate
               : m_defaults.historyWindowSize;
  }

  [[nodiscard]] auto budgetActivationLaunches() const -> std::uint64_t {
    if (m_defaults.queue && !m_defaults.queue->disable)
      return static_cast<std::uint64_t>(m_defaults.queue->maxConsecutiveRuns);
    return 1u;
  }

  [[nodiscard]] auto budgetEstimatedCost(std::size_t candidate) const
      -> double {
    auto cost = m_budget->costEstimate;
    if (m_budgetIncumbent && !m_histories[candidate].empty())
      cost += std::max(0.0, m_budgetStatistics[candidate].median -
                                m_budgetStatistics[*m_budgetIncumbent].median);
    return cost + m_budget->policy.measurementCostHintSeconds.value_or(0.0);
  }

  [[nodiscard]] auto budgetCandidateGain(std::size_t candidate) const
      -> double {
    auto const &best = m_budgetStatistics[*m_budgetIncumbent];
    if (!m_budgetEpochFresh[candidate]) {
      auto prior = m_budgetStatistics[candidate];
      prior.acceptedSampleCount = 1u;
      prior.standardDeviation = std::max(best.median, prior.median) * 0.5;
      return detail::BenefitBudget::expectedGain(best, prior);
    }
    return m_histories[candidate].empty()
               ? m_budget->unseenGain(best.median)
               : detail::BenefitBudget::expectedGain(
                     best, m_budgetStatistics[candidate]);
  }

  [[nodiscard]] auto budgetCandidateWorthwhile(std::size_t candidate) -> bool {
    ensureBudgetStorage();
    if (m_budgetInferior[candidate] ||
        (!m_histories[candidate].empty() &&
         m_budgetVisitSamples[candidate] >= budgetSampleLimit()))
      return false;
    auto const activation = budgetActivationLaunches();
    return m_budget->worthwhile(budgetCandidateGain(candidate),
                                budgetEstimatedCost(candidate) *
                                    static_cast<double>(activation),
                                activation);
  }

  void finishBudgetExploration(TunerCompletionReason reason) {
    m_budgetStoppingReason = reason;
    m_budgetDirectCandidate.reset();
    if (!m_explorationComplete)
      finishTuningAtBudget(reason);
  }

  void retireBudgetCandidate(std::size_t candidate) {
    if (m_queue)
      static_cast<void>(m_queue->retire(candidate));
    if (m_budgetDirectCandidate == candidate)
      m_budgetDirectCandidate.reset();
    if (m_scheduled[candidate]) {
      m_scheduled[candidate] = false;
      --m_scheduledCount;
    }
  }

  /** A queued activation respects warmups and burst interleaving. */
  auto selectBudgetActivation() -> detail::CandidateQueue::Selection {
    auto selected = m_queue ? *m_queue->next()
                            : detail::CandidateQueue::Selection{
                                  *m_budgetDirectCandidate, false, true, true};
    auto const candidate = selected.candidateIndex;
    if (selected.beginActivation)
      m_budgetWarmups[candidate] = m_defaults.queue->warmupRuns;
    if (m_budgetColdStart && !m_budgetIncumbent && m_budget->remaining <= 1u)
      m_budgetWarmups[candidate] = 0u;
    if (m_budgetWarmups[candidate] > 0u) {
      --m_budgetWarmups[candidate];
      selected.measure = false;
      m_budgetPurpose = LaunchPurpose::warmup;
    } else
      m_budgetPurpose = LaunchPurpose::experiment;
    return selected;
  }

  auto nextBudgetCandidate() -> detail::CandidateQueue::Selection {
    ensureBudgetStorage();
    auto active = [&] {
      return m_queue ? !m_queue->empty() : m_budgetDirectCandidate.has_value();
    };

    if (m_budgetIncumbent && !m_explorationComplete && executionBudgetReached())
      finishBudgetExploration(TunerCompletionReason::maximumExecutions);
    if (m_budgetIncumbent && !m_explorationComplete &&
        retiredConfigurationBudgetReached())
      finishBudgetExploration(
          TunerCompletionReason::maximumRetiredConfigurations);
    if (m_budgetIncumbent && !m_explorationComplete &&
        m_budget->remaining == 0u)
      finishBudgetExploration(
          TunerCompletionReason::insufficientExpectedBenefit);
    if (m_budgetColdStart && m_budgetBootstrapSamples < 3u && active() &&
        !m_explorationComplete)
      return selectBudgetActivation();
    if (m_budgetIncumbent && !m_explorationComplete &&
        !m_budget->affordable(
            m_budget->costEstimate +
            m_budget->policy.measurementCostHintSeconds.value_or(0.0)))
      finishBudgetExploration(TunerCompletionReason::tuningOverheadBudget);

    // Health checks run only during adaptive reuse and never reopen
    // exploration.
    if (m_explorationComplete) {
      if (!m_budgetIncumbent)
        throw std::logic_error{
            "No valid measured configuration is available to launch."};
      if (m_defaults.selection == SelectionPolicy::adaptive &&
          m_budget->launches >= m_budgetNextCheck) {
        m_budgetNextCheck = m_budget->launches + m_budget->policy.checkInterval;
        if (m_budget->affordable(budgetEstimatedCost(*m_budgetIncumbent))) {
          m_budgetPurpose = LaunchPurpose::healthCheck;
          return {*m_budgetIncumbent, true, true, true};
        }
      }
      if (m_defaults.selection == SelectionPolicy::adaptive &&
          m_budgetEpochChanged) {
        auto challenger = std::optional<std::size_t>{};
        auto value = 0.0;
        for (std::size_t candidate = 0u; candidate < m_candidateCount;
             ++candidate)
          if (candidate != *m_budgetIncumbent && candidateUsable(candidate) &&
              !m_rejected[candidate] && !m_histories[candidate].empty() &&
              budgetCandidateWorthwhile(candidate)) {
            auto const gain = budgetCandidateGain(candidate);
            if (gain > value) {
              value = gain;
              challenger = candidate;
            }
          }
        if (challenger) {
          m_budgetPurpose = LaunchPurpose::experiment;
          return {*challenger, true, true, true};
        }
        m_budgetEpochChanged = false;
      }
      m_budgetPurpose = LaunchPurpose::production;
      return {*m_budgetIncumbent, false, false, false};
    }
    if (active()) {
      if (m_queue && !m_queue->full() && m_budgetIncumbent &&
          m_budget->worthwhile(
              m_budget->unseenGain(
                  m_budgetStatistics[*m_budgetIncumbent].median),
              (m_budget->costEstimate +
               m_budget->policy.measurementCostHintSeconds.value_or(0.0)) *
                  static_cast<double>(budgetActivationLaunches()),
              budgetActivationLaunches())) {
        auto proposal = recommendCandidate();
        if (proposal &&
            proposal->disposition == RecommendationDisposition::scheduled)
          static_cast<void>(m_queue->insert(proposal->candidate));
      }
      auto selected = selectBudgetActivation();
      if (!m_budgetIncumbent ||
          budgetCandidateWorthwhile(selected.candidateIndex))
        return selected;
      retireBudgetCandidate(selected.candidateIndex);
      // Retry at most once per resident slot; no strategy work on production.
      while (active()) {
        selected = selectBudgetActivation();
        if (budgetCandidateWorthwhile(selected.candidateIndex))
          return selected;
        retireBudgetCandidate(selected.candidateIndex);
      }
    }
    // Resolve measured uncertainty before spending on another proposal.
    auto challenger = std::optional<std::size_t>{};
    auto bestGain = 0.0;
    if (m_budgetIncumbent)
      for (std::size_t candidate = 0u; candidate < m_candidateCount;
           ++candidate)
        if (candidate != *m_budgetIncumbent && candidateUsable(candidate) &&
            !m_rejected[candidate] && !m_histories[candidate].empty() &&
            budgetCandidateWorthwhile(candidate)) {
          auto const gain = budgetCandidateGain(candidate);
          if (gain > bestGain) {
            bestGain = gain;
            challenger = candidate;
          }
        }
    if (challenger) {
      m_histories[*challenger].reopen();
      if (m_queue)
        static_cast<void>(m_queue->insert(*challenger));
      else
        m_budgetDirectCandidate = challenger;
      if (!m_scheduled[*challenger]) {
        m_scheduled[*challenger] = true;
        ++m_scheduledCount;
      }
      return selectBudgetActivation();
    }
    if (m_budgetIncumbent &&
        !m_budget->worthwhile(
            m_budget->unseenGain(m_budgetStatistics[*m_budgetIncumbent].median),
            (m_budget->costEstimate +
             m_budget->policy.measurementCostHintSeconds.value_or(0.0)) *
                static_cast<double>(budgetActivationLaunches()),
            budgetActivationLaunches())) {
      finishBudgetExploration(
          TunerCompletionReason::insufficientExpectedBenefit);
      return nextBudgetCandidate();
    }
    applySpaceFeedback();
    ensureBudgetStorage();
    if (m_explorationComplete)
      return nextBudgetCandidate();
    for (std::size_t attempt = 0u;
         attempt < m_defaults.maximumConsecutiveStrategyRetries; ++attempt) {
      auto const proposal = recommendCandidate();
      if (proposal &&
          proposal->disposition == RecommendationDisposition::scheduled) {
        if (m_queue)
          static_cast<void>(m_queue->insert(proposal->candidate));
        else
          m_budgetDirectCandidate = proposal->candidate;
        return selectBudgetActivation();
      }
      // Proposal processing itself is charged by the caller; stop when all
      // candidates have resolved rather than synthesizing a different proposal.
      if (m_defaults.selection == SelectionPolicy::fixed &&
          allFixedCandidatesResolved()) {
        finishTuning();
        return nextBudgetCandidate();
      }
    }
    if (!m_budgetIncumbent)
      throw std::invalid_argument{"No valid configuration could be admitted."};
    finishBudgetExploration(TunerCompletionReason::insufficientExpectedBenefit);
    return nextBudgetCandidate();
  }

  void recordBudgetMetric(std::size_t candidate, double value) {
    ensureBudgetStorage();
    auto const oldBest = m_budgetIncumbent
                             ? m_budgetStatistics[*m_budgetIncumbent].median
                             : std::numeric_limits<double>::infinity();
    auto &history = m_histories[candidate];
    if (!m_budgetEpochFresh[candidate]) {
      auto samples = std::array{value};
      history.restoreCompleted(samples);
      m_budgetEpochFresh[candidate] = true;
    } else {
      history.reopen();
      history.beginActivation(false);
      static_cast<void>(history.record(value));
      history.retire(detail::RuntimeCompletion::adaptiveVisit);
    }
    m_budgetStatistics[candidate] = history.statistics();
    ++m_budget->measurements;
    ++m_budgetVisitSamples[candidate];
    if (m_budgetColdStart && m_budgetBootstrapSamples < 3u)
      ++m_budgetBootstrapSamples;
    if (m_budgetPurpose == LaunchPurpose::healthCheck) {
      auto const &reference = m_budgetCheckReference;
      auto const low =
          reference.acceptedSampleCount >= 5u
              ? reference.confidenceLow
              : reference.median -
                    2.576 * detail::BenefitBudget::uncertainty(reference);
      auto const high =
          reference.acceptedSampleCount >= 5u
              ? reference.confidenceHigh
              : reference.median +
                    2.576 * detail::BenefitBudget::uncertainty(reference);
      m_budgetDriftSamples =
          value < low || value > high ? m_budgetDriftSamples + 1u : 0u;
      if (m_budgetDriftSamples >= 2u) {
        auto samples = std::array{m_budgetLastDriftValue, value};
        history.restoreCompleted(samples);
        m_budgetStatistics[candidate] = history.statistics();
        std::fill(m_budgetEpochFresh.begin(), m_budgetEpochFresh.end(), false);
        m_budgetEpochFresh[candidate] = true;
        std::fill(m_budgetInferior.begin(), m_budgetInferior.end(), false);
        std::fill(m_budgetVisitSamples.begin(), m_budgetVisitSamples.end(), 0u);
        m_budgetEpochChanged = true;
        m_budget->unimprovedProposals = 0u;
        m_budgetDriftSamples = 0u;
        m_budgetCheckReference = m_budgetStatistics[candidate];
      }
      m_budgetLastDriftValue = value;
    }
    refreshBudgetIncumbent();
    if (m_budgetStatistics[*m_budgetIncumbent].median < oldBest)
      m_budget->unimprovedProposals = 0u;
    else if (m_budgetPurpose == LaunchPurpose::experiment &&
             candidate != *m_budgetIncumbent)
      ++m_budget->unimprovedProposals;
    if (m_defaults.mannWhitneyEarlyStop && candidate != *m_budgetIncumbent &&
        detail::mannWhitneyUCompare(history, m_histories[*m_budgetIncumbent],
                                    m_defaults.mannWhitneyMinimumSamples,
                                    m_defaults.mannWhitneyAlpha) ==
            RuntimeComparison::slower)
      m_budgetInferior[candidate] = true;
    auto const bootstrapFinished =
        m_budgetColdStart && m_budgetBootstrapSamples >= 3u;
    if (bootstrapFinished || m_budgetInferior[candidate] ||
        (!m_budgetColdStart &&
         m_budgetVisitSamples[candidate] >= budgetSampleLimit())) {
      retireBudgetCandidate(candidate);
      if (!m_explorationComplete) {
        recordRetiredConfiguration(candidate);
        m_refinementPending =
            m_space.observe(candidate, history.statistics().median) ||
            m_refinementPending;
      }
      if (bootstrapFinished) {
        m_budgetColdStart = false;
        m_budgetCheckReference = m_budgetStatistics[*m_budgetIncumbent];
      }
    }
    if (m_budgetCheckReference.acceptedSampleCount == 0u)
      m_budgetCheckReference = m_budgetStatistics[*m_budgetIncumbent];
    stageCache(candidate, true);
  }

  template <typename Queue, typename Spec, typename Bundle>
  void bindBudgetContext(Queue const &queue, Spec const &spec,
                         Bundle const &bundle) {
    static constexpr char tag{};
    if (!(queue.getDevice() == m_device))
      throw std::invalid_argument{
          "The queue device does not match this tuner's device."};
    if (m_budgetContextTag) {
      if (m_budgetContextTag != &tag)
        throw std::logic_error{
            "A tuner is bound to one kernel and one launch configuration."};
      m_budgetContextValidator(&spec);
      return;
    }
    validatePrototype(bundle);
    bindFingerprint(makeFingerprint(spec, bundle),
                    detail::typeName<typename Bundle::KernelFn>(),
                    launchDescription(spec));
    m_budgetContextValidator = [original = spec](void const *raw) {
      auto const &current = *static_cast<Spec const *>(raw);
      auto same = false;
      if constexpr (alpaka::onHost::isFrameSpec_v<Spec>)
        same = original.getNumFrames() == current.getNumFrames() &&
               original.getFrameExtents() == current.getFrameExtents();
      else
        same = original.getNumBlocks() == current.getNumBlocks() &&
               original.getNumThreads() == current.getNumThreads();
      if (!same)
        throw std::logic_error{
            "A tuner is bound to one kernel and one launch configuration."};
    };
    bindLaunchValidator<Queue>(spec, bundle);
    initialiseCompileVariants<Queue, Spec, Bundle>();
    initialiseScheduling();
    ensureBudgetStorage();
    m_budgetColdStart = !currentBestCandidate().has_value();
    refreshBudgetIncumbent();
    m_budgetNextCheck = m_budget->policy.checkInterval;
    if (m_budgetIncumbent)
      m_budgetCheckReference = m_budgetStatistics[*m_budgetIncumbent];
    m_budgetContextTag = &tag;
  }

  template <typename Queue, typename Spec, typename Kernel, typename... Args>
  auto enqueueBudgeted(Queue const &queue, Spec const &spec,
                       alpaka::KernelBundle<Kernel, Args...> const &bundle,
                       bool observed = true) -> LaunchObservation {
    auto const production =
        m_budgetContextTag && m_explorationComplete && m_budgetIncumbent &&
        !m_budgetEpochChanged &&
        (m_defaults.selection == SelectionPolicy::fixed ||
         m_budget->launches < m_budgetNextCheck) &&
        m_consumedValidityGeneration == *m_validityGeneration &&
        (!m_pendingMetric || !m_pendingMetric->measure);
    auto const start = production ? std::chrono::steady_clock::time_point{}
                                  : std::chrono::steady_clock::now();
    m_recommendationSecondsSinceLastLaunch = 0.0;
    bindBudgetContext(queue, spec, bundle);
    reconcileUserInvalidations();
    reconcileMissingMetric();
    if (!production)
      bindLaunchValidator<Queue>(spec, bundle);
    auto const selected =
        production ? detail::CandidateQueue::Selection{*m_budgetIncumbent,
                                                       false, false, false}
                   : nextBudgetCandidate();
    if (production)
      m_budgetPurpose = LaunchPurpose::production;
    if constexpr (!usesCustomMetric &&
                  !std::same_as<ALPAKA_TYPEOF(queue.getTiming()),
                                alpaka::timing::Enabled>)
      if (selected.measure)
        throw std::invalid_argument{
            "A measured tuning launch requires a timing-enabled queue."};
    auto const hadIncumbent = m_budgetIncumbent.has_value();
    auto const incumbentSeconds =
        m_budgetIncumbent ? m_budgetStatistics[*m_budgetIncumbent].median : 0.0;
    using Bundle = alpaka::KernelBundle<Kernel, Args...>;
    if (m_budgetDispatchCandidate != selected.candidateIndex) {
      auto const key = compileVariantKey(indicesFor(selected.candidateIndex));
      m_budgetCachedDispatch = m_compileVariantIndices.at(key);
      m_budgetDispatchCandidate = selected.candidateIndex;
    }
    LaunchCall<Queue, Spec, Bundle> call{this, &queue, &spec, &bundle,
                                         std::nullopt};
    m_compileVariants[m_budgetCachedDispatch](
        &call, selected.candidateIndex, selected.measure,
        selected.beginActivation, selected.endActivation, true);
    m_budget->launched();
    if (m_budgetPurpose == LaunchPurpose::production)
      ++m_budget->productionLaunches;
    auto const execution =
        recordExecution(selected.candidateIndex, call.runtimeSeconds);
    if constexpr (usesCustomMetric)
      m_pendingMetric =
          PendingMetric{execution, selected.candidateIndex, selected.measure,
                        selected.beginActivation, selected.endActivation};
    if (!production) {
      auto const wall = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - start)
                            .count();
      auto const hostCost =
          std::max(0.0, wall - call.runtimeSeconds.value_or(0.0));
      m_budgetPendingHostCost = hostCost;
      m_budgetPendingIncumbent = hadIncumbent
                                     ? incumbentSeconds
                                     : std::numeric_limits<double>::infinity();
      if constexpr (!usesCustomMetric) {
        auto const cost =
            hostCost +
            (call.runtimeSeconds && hadIncumbent
                 ? std::max(0.0, *call.runtimeSeconds - incumbentSeconds)
                 : 0.0) +
            (selected.measure
                 ? m_budget->policy.measurementCostHintSeconds.value_or(0.0)
                 : 0.0);
        m_budget->charge(cost);
        if (selected.measure)
          m_budget->observeCost(hostCost);
      } else if (!selected.measure)
        m_budget->charge(hostCost);
    }
    if (!observed)
      return {};
    return LaunchObservation{
        .purpose = m_budgetPurpose,
        .candidateIndex = selected.candidateIndex,
        .configuration = normalizedConfiguration(selected.candidateIndex),
        .runtimeSeconds = call.runtimeSeconds,
        .metricValue = call.runtimeSeconds,
        .runtimeMeasurementSource = runtimeMeasurementSource(),
        .recommendationSeconds = m_recommendationSecondsSinceLastLaunch,
        .measured = call.runtimeSeconds.has_value(),
        .tuningComplete = m_explorationComplete,
        .loadedFromCache = m_loadedFromCache};
  }

  void requirePendingMetric() const {
    if (!m_pendingMetric || m_executionHistory.empty() ||
        m_pendingMetric->executionIndex != m_executionHistory.size() - 1u)
      throw std::logic_error{
          "The most recent enqueue is not awaiting a custom metric."};
  }

  template <typename Queue, typename FrameSpec, typename Kernel,
            typename... Args>
  [[nodiscard]] auto
  tryFastReplay(Queue const &queue, FrameSpec const &frameSpec,
                alpaka::KernelBundle<Kernel, Args...> const &prototype)
      -> bool {
    if (!(queue.getDevice() == m_device))
      throw std::invalid_argument{
          "The queue device does not match this tuner's device."};
    validatePrototype(prototype);
    auto const fingerprint =
        makeFingerprint<FrameSpec, decltype(prototype)>(frameSpec, prototype);
    bindFingerprint(
        fingerprint,
        detail::typeName<
            typename std::remove_cvref_t<decltype(prototype)>::KernelFn>(),
        launchDescription(frameSpec));
    bindLaunchValidator<Queue>(frameSpec, prototype);
    initialiseScheduling();
    reconcileUserInvalidations();
    reconcileMissingMetric();
    if (!m_terminal)
      return false;
    if (m_bestCandidate == std::numeric_limits<std::size_t>::max())
      throw std::logic_error{
          "The tuner terminated without a measured candidate to launch."};

    using Bundle = std::remove_cvref_t<decltype(prototype)>;
    initialiseCompileVariants<Queue, FrameSpec, Bundle>();
    auto const candidate = m_bestCandidate;
    LaunchCall<Queue, FrameSpec, Bundle> call{.tuner = this,
                                              .queue = &queue,
                                              .frameSpec = &frameSpec,
                                              .prototype = &prototype,
                                              .runtimeSeconds = std::nullopt};
    auto const key = compileVariantKey(indicesFor(candidate));
    m_compileVariants.at(m_compileVariantIndices.at(key))(
        &call, candidate, false, false, false, false);
    auto const executionIndex = recordExecution(candidate, call.runtimeSeconds);
    if constexpr (usesCustomMetric)
      m_pendingMetric = PendingMetric{.executionIndex = executionIndex,
                                      .candidateIndex = candidate,
                                      .measure = false,
                                      .beginActivation = false,
                                      .endActivation = false};
    return true;
  }

  template <typename Bundle>
  void validatePrototype(Bundle const &prototype) const {
    std::vector<std::string_view> markers;
    alpaka::apply(
        [&markers](auto const &...arguments) {
          (collectPrototypeMarker<decltype(arguments)>(markers), ...);
        },
        prototype.getArgs());
    std::apply(
        [&markers](auto const &...entries) {
          (validateTunableConsumed<decltype(entries)>(markers), ...);
        },
        m_tunables.entries());
  }

  template <typename Argument>
  static void collectPrototypeMarker(std::vector<std::string_view> &markers) {
    using CleanArgument = std::remove_cvref_t<Argument>;
    if constexpr (detail::isMarkedTunable<CleanArgument>)
      markers.push_back(CleanArgument::name.view());
  }

  template <typename Entry>
  static void
  validateTunableConsumed(std::vector<std::string_view> const &markers) {
    using CleanEntry = std::remove_cvref_t<Entry>;
    auto const consumed = detail::isReservedLaunchName<CleanEntry::name> ||
                          std::find(markers.begin(), markers.end(),
                                    CleanEntry::nameView()) != markers.end();
    if (!consumed)
      throw std::invalid_argument{
          "Every declared tunable must occur in the KernelBundle or be a "
          "reserved launch tunable."};
  }

  template <FixedString Name, typename Arguments, std::size_t Index = 0u>
  static consteval auto containsMarker() -> bool {
    if constexpr (Index == std::tuple_size_v<Arguments>)
      return false;
    else {
      using Argument = std::tuple_element_t<Index, Arguments>;
      if constexpr (detail::isMarkedTunable<Argument> &&
                    detail::sameName<Name, Argument::name>)
        return true;
      return containsMarker<Name, Arguments, Index + 1u>();
    }
  }

  void initialiseDimensions() {
    m_dimensionSizes.reserve(dimensionCount);
    std::apply(
        [this](auto const &...entries) {
          (appendEntryDimensions(entries), ...);
        },
        m_tunables.entries());
    m_space.initialise(m_dimensionSizes, Traits::generatesCandidates,
                       Traits::enumerable, m_defaults.space, m_baseRandomSeed);
    m_candidateCount = m_space.size();
    initialiseDependencies();
    auto preferred = CandidateIndices{};
    auto categorical = std::array<bool, dimensionCount>{};
    bool hasPreferred{};
    std::apply(
        [&](auto const &...entries) {
          (
              [&] {
                using Entry = std::remove_cvref_t<decltype(entries)>;
                using Values = typename Entry::values_type;
                if constexpr (detail::isAutomaticCandidates<Values>) {
                  constexpr auto offset =
                      Traits::template dimensionOffset<Entry::name>;
                  constexpr auto arity =
                      detail::candidateDimensionCount<Values>;
                  for (std::size_t d{}; d < arity; ++d)
                    categorical[offset + d] = entries.values.categorical();
                  if constexpr (arity == 1u)
                    if (auto value = entries.values.preferredIndex()) {
                      preferred[offset] = *value;
                      hasPreferred = true;
                    }
                }
              }(),
              ...);
        },
        m_tunables.entries());
    m_space.setHints(preferred, hasPreferred, categorical);
  }

  using CandidateIndices = std::array<std::size_t, dimensionCount>;

  template <typename Names, std::size_t Parent = 0u, typename Function,
            typename... Args>
  auto invokeParents(CandidateIndices const &indices, Function &&function,
                     Args const &...args) const -> bool {
    if constexpr (Parent == std::tuple_size_v<Names>)
      return static_cast<bool>(
          std::invoke(std::forward<Function>(function), args...));
    else {
      using Name = std::tuple_element_t<Parent, Names>;
      static_assert(Traits::template has<Name::name>,
                    "A dependent domain refers to an undeclared parent.");
      return withCandidateValue<Name::name>(indices, [&](auto const &value) {
        return invokeParents<Names, Parent + 1u>(
            indices, std::forward<Function>(function), args..., value);
      });
    }
  }

  template <typename Entry>
  void appendDomainProjector(Entry const &entry,
                             std::vector<std::vector<std::size_t>> &parents) {
    using Values = typename Entry::values_type;
    constexpr auto index = Traits::template index<Entry::name>;
    constexpr auto offset = Traits::template dimensionOffset<Entry::name>;
    constexpr auto arity = detail::candidateDimensionCount<Values>;
    if constexpr (requires { typename Values::parent_names; }) {
      using Names = typename Values::parent_names;
      [&]<std::size_t... I>(std::index_sequence<I...>) {
        (parents[index].push_back(
             Traits::template index<std::tuple_element_t<I, Names>::name>),
         ...);
      }(std::make_index_sequence<std::tuple_size_v<Names>>{});
    }
    m_domainProjectors.emplace_back([](Tuner const &self,
                                       CandidateIndices &indices,
                                       bool sampleGlobal) {
      auto const &entry = std::get<Traits::template index<Entry::name>>(
          self.m_tunables.entries());
      if constexpr (detail::isAutomaticCandidates<Values> && arity == 1u) {
        auto projected = entry.values.project(indices[offset], sampleGlobal);
        if (!projected)
          return false;
        indices[offset] = *projected;
      }
      if constexpr (detail::isAutomaticCandidates<Values>) {
        if (entry.values.alignment() > 1u) {
          auto aligned = self.template withCandidateValue<Entry::name>(
              indices, [&](auto const &value) {
                auto check = [&](auto const &scalar) {
                  using Scalar = std::remove_cvref_t<decltype(scalar)>;
                  if constexpr (std::integral<Scalar>)
                    return static_cast<long double>(scalar) /
                               entry.values.alignment() ==
                           std::floor(static_cast<long double>(scalar) /
                                      entry.values.alignment());
                  else
                    return false;
                };
                if constexpr (alpaka::isVector_v<
                                  std::remove_cvref_t<decltype(value)>>) {
                  for (std::size_t d{}; d < arity; ++d)
                    if (!check(value[d]))
                      return false;
                  return true;
                } else
                  return check(value);
              });
          if (!aligned)
            return false;
        }
      }
      if constexpr (requires { typename Values::parent_names; }) {
        static_assert(
            arity == 1u && detail::isRVals<Values>,
            "Dependent domains currently require a scalar runtime child.");
        using Names = typename Values::parent_names;
        return self.template invokeParents<Names>(
            indices, [&](auto const &...values) {
              auto const allowed =
                  std::invoke(entry.values.generator, values...);
              auto const selected = entry.values.at(indices[offset]);
              if (allowed.indexOf(selected))
                return true;
              if (allowed.size() == 0u)
                return false;
              auto const replacement = entry.values.indexOf(
                  allowed.at(indices[offset] % allowed.size()));
              if (!replacement)
                return false;
              auto const projected = entry.values.project(*replacement);
              if (!projected || !allowed.indexOf(entry.values.at(*projected)))
                return false;
              indices[offset] = *projected;
              return true;
            });
      }
      return true;
    });
  }

  void initialiseDependencies() {
    auto parents = std::vector<std::vector<std::size_t>>(Traits::size);
    std::apply(
        [&](auto const &...entries) {
          (appendDomainProjector(entries, parents), ...);
        },
        m_tunables.entries());
    auto unsorted = std::move(m_domainProjectors);
    auto marks = std::vector<unsigned>(Traits::size);
    auto visit = [&](auto &&self, std::size_t index) -> void {
      if (marks[index] == 2u)
        return;
      if (marks[index] == 1u)
        throw std::invalid_argument{"Dependent tuning domains form a cycle."};
      marks[index] = 1u;
      for (auto parent : parents[index])
        self(self, parent);
      marks[index] = 2u;
      m_domainProjectors.push_back(std::move(unsorted[index]));
    };
    for (std::size_t index{}; index < Traits::size; ++index)
      visit(visit, index);
  }

  void resizeCandidateStorage() {
    m_candidateCount = m_space.size();
    m_histories.resize(m_candidateCount,
                       detail::RuntimeHistory{runtimeHistoryOptions()});
    m_scheduled.resize(m_candidateCount, false);
    m_rejected.resize(m_candidateCount, false);
    m_userInvalidated.resize(m_candidateCount, false);
    m_missingMetric.resize(m_candidateCount, false);
    m_loadedFromHistoryCandidates.resize(m_candidateCount, false);
    while (m_candidateValidity.size() < m_candidateCount)
      m_candidateValidity.push_back(makeValidityState());
  }

  void expandSpace() {
    auto ranked = std::vector<std::pair<double, std::size_t>>{};
    for (std::size_t id{}; id < m_histories.size(); ++id)
      if (candidateUsable(id) && !m_histories[id].empty())
        ranked.emplace_back(m_histories[id].statistics().estimate(), id);
    std::ranges::sort(ranked);
    auto best = std::vector<std::size_t>{};
    for (std::size_t i{}; i < std::min<std::size_t>(4u, ranked.size()); ++i)
      best.push_back(ranked[i].second);
    m_space.expand(
        [this](CandidateIndices &indices, bool sampleGlobal) {
          for (auto const &project : m_domainProjectors)
            if (!project(*this, indices, sampleGlobal))
              return false;
          return true;
        },
        [this](CandidateIndices const &indices) {
          return candidateAccepted(indices) &&
                 (!m_launchValidator || m_launchValidator(*this, indices));
        },
        best);
    resizeCandidateStorage();
  }

  [[nodiscard]] auto finishResolvedCatalog() -> bool {
    if (m_explorationComplete || !m_space.usesCandidateCatalog() ||
        m_space.canExpand())
      return false;
    for (std::size_t candidate = 0u; candidate < m_candidateCount; ++candidate)
      if (!m_rejected.at(candidate) && candidateUsable(candidate) &&
          (m_histories.at(candidate).empty() ||
           (m_defaults.selection == SelectionPolicy::fixed &&
            !m_histories.at(candidate).isFinished())))
        return false;
    finishTuning();
    return true;
  }

  void applySpaceFeedback() {
    if (!m_refinementPending || m_explorationComplete ||
        m_defaults.exploration == ExplorationPolicy::offline)
      return;
    m_refinementPending = false;
    expandSpace();
    static_cast<void>(finishResolvedCatalog());
  }

  void replenishResolvedSpace() {
    if (!m_space.canExpand() || m_explorationComplete ||
        m_defaults.exploration == ExplorationPolicy::offline)
      return;
    for (std::size_t id{}; id < m_candidateCount; ++id)
      if (!m_rejected[id] && candidateUsable(id) &&
          !m_histories[id].isFinished())
        return;
    expandSpace();
  }

  [[nodiscard]] auto spaceCompletionReason() const -> TunerCompletionReason {
    switch (m_space.state()) {
    case SpaceState::candidateBudget:
      return TunerCompletionReason::candidateBudget;
    case SpaceState::plateau:
      return TunerCompletionReason::plateau;
    case SpaceState::stalled:
      return TunerCompletionReason::generationStalled;
    default:
      return TunerCompletionReason::allConfigurations;
    }
  }

  template <typename Value>
  [[nodiscard]] static auto learnedNumericValue(Value const &value,
                                                std::size_t component)
      -> std::optional<float> {
    using Type = std::remove_cvref_t<Value>;
    auto convert = [](auto const &scalar) -> std::optional<float> {
      using Scalar = std::remove_cvref_t<decltype(scalar)>;
      if constexpr (std::is_arithmetic_v<Scalar> || std::is_enum_v<Scalar>) {
        auto const converted = static_cast<float>(scalar);
        if (std::isfinite(converted))
          return converted;
      }
      return std::nullopt;
    };
    if constexpr (alpaka::isVector_v<Type>)
      return convert(value[component]);
    else {
      static_cast<void>(component);
      return convert(value);
    }
  }

  template <typename Entry>
  void appendLearnedDimensions(LearnedModelContextDescriptor &descriptor,
                               std::size_t &offset, Entry const &entry) const {
    using Values = typename Entry::values_type;
    constexpr auto arity = detail::candidateDimensionCount<Values>;
    auto kind = [] {
      if constexpr (detail::isReservedLaunchName<Entry::name>)
        return LearnedDimensionKind::launch;
      else if constexpr (detail::isRVals<Values>)
        return LearnedDimensionKind::runtime;
      else if constexpr (detail::isCVals<Values>)
        return LearnedDimensionKind::compileTime;
      else
        return LearnedDimensionKind::categorical;
    }();
    if constexpr (detail::isAutomaticCandidates<Values> &&
                  !detail::isReservedLaunchName<Entry::name>)
      if (entry.values.categorical())
        kind = LearnedDimensionKind::categorical;
    for (std::size_t component = 0u; component < arity; ++component) {
      auto dimension = LearnedDimensionDescriptor{
          .name = std::string{Entry::nameView()},
          .kind = kind,
          .cardinality = m_dimensionSizes.at(offset + component),
          .componentIndex = component,
          .vectorArity = arity,
          .concreteValues = {}};
      if (dimension.cardinality <= 4096u)
        dimension.concreteValues.reserve(dimension.cardinality);
      for (std::size_t candidate = 0u;
           candidate <
           (dimension.cardinality <= 4096u ? dimension.cardinality : 0u);
           ++candidate) {
        auto indices = std::array<std::size_t, dimensionCount>{};
        indices[offset + component] = candidate;
        auto numeric = std::optional<float>{};
        static_cast<void>(withCandidateValue<Entry::name>(
            indices, [&numeric, component](auto const &value) {
              numeric = learnedNumericValue(value, component);
              return true;
            }));
        if (!numeric) {
          dimension.concreteValues.clear();
          break;
        }
        dimension.concreteValues.push_back(*numeric);
      }
      descriptor.dimensions.push_back(std::move(dimension));
    }
    offset += arity;
  }

  static void
  appendLearnedHashFeatures(LearnedModelContextDescriptor &descriptor,
                            std::string_view prefix, std::string_view value) {
    auto const hashes = learnedSignedHashFeatures(value, 8u);
    for (std::size_t index = 0u; index < hashes.size(); ++index)
      descriptor.contextFeatures.push_back(
          {std::string{prefix} + "_hash_" + std::to_string(index),
           hashes[index]});
  }

  [[nodiscard]] auto learnedModelContext() const
      -> LearnedModelContextDescriptor {
    auto descriptor = LearnedModelContextDescriptor{};
    descriptor.schemaVersion = m_space.usesCandidateCatalog() ? 2u : 1u;
    auto const properties = m_device.getDeviceProperties();
    descriptor.deviceClass = properties.warpSize > 1u ? LearnedDeviceClass::gpu
                                                      : LearnedDeviceClass::cpu;
    auto domainSizeFeature = std::log1p(static_cast<float>(m_candidateCount));
    if (m_space.usesCandidateCatalog()) {
      domainSizeFeature = 0.0f;
      for (auto size : m_dimensionSizes)
        domainSizeFeature += std::log(static_cast<float>(size));
    }
    descriptor.contextFeatures = {
        {"candidate_count_log1p", domainSizeFeature},
        {"dimension_count", static_cast<float>(dimensionCount)},
        {"multiprocessor_count",
         static_cast<float>(properties.multiProcessorCount)},
        {"warp_size", static_cast<float>(properties.warpSize)},
        {"max_threads_per_block",
         static_cast<float>(properties.maxThreadsPerBlock)},
        {"max_blocks_per_grid_log1p",
         std::log1p(static_cast<float>(properties.maxBlocksPerGrid))},
        {"global_memory_bytes_log1p",
         std::log1p(static_cast<float>(properties.globalMemCapacityBytes))},
        {"shared_memory_per_block_bytes_log1p",
         std::log1p(static_cast<float>(properties.sharedMemPerBlockBytes))},
    };
    auto identity = std::string{};
    for (auto const &entry : m_identityEntries) {
      identity += entry;
      identity.push_back('\n');
    }
    appendLearnedHashFeatures(descriptor, "kernel", m_kernelName);
    appendLearnedHashFeatures(descriptor, "launch", m_launchSpecification);
    appendLearnedHashFeatures(descriptor, "context", identity);
    appendLearnedHashFeatures(descriptor, "device", properties.getName());
    auto offset = std::size_t{};
    std::apply(
        [this, &descriptor, &offset](auto const &...entries) {
          (appendLearnedDimensions(descriptor, offset, entries), ...);
        },
        m_tunables.entries());
    if (m_defaults.strategy == StrategyKind::learnedHybrid) {
      if (!m_space.usesCandidateCatalog())
        descriptor.legalCandidates.resize(m_candidateCount);
      for (std::size_t candidate = 0u;
           candidate < descriptor.legalCandidates.size(); ++candidate)
        descriptor.legalCandidates[candidate] =
            static_cast<std::uint8_t>(candidateUsable(candidate) &&
                                      candidateAccepted(indicesFor(candidate)));
    }
    return descriptor;
  }

  template <typename Entry> void appendEntryDimensions(Entry const &entry) {
    using Values = typename Entry::values_type;
    if constexpr (detail::isRVals<Values>) {
      if constexpr (alpaka::isVector_v<typename Values::value_type>) {
        appendRuntimeVectorDimensions(
            entry, std::make_index_sequence<
                       detail::candidateDimensionCount<Values>>{});
      } else {
        m_dimensionSizes.push_back(entrySize(entry));
      }
    } else if constexpr (detail::isCTypes<Values> &&
                         detail::candidateDimensionCount<Values> > 1u) {
      appendCompileVectorDimensions<Values>(
          std::make_index_sequence<detail::candidateDimensionCount<Values>>{});
    } else {
      m_dimensionSizes.push_back(entrySize(entry));
    }
  }

  template <typename Entry, std::size_t... Dimension>
  void appendRuntimeVectorDimensions(Entry const &entry,
                                     std::index_sequence<Dimension...>) {
    (m_dimensionSizes.push_back(runtimeUniqueComponentCount(entry, Dimension)),
     ...);
  }

  template <typename Values, std::size_t... Dimension>
  void appendCompileVectorDimensions(std::index_sequence<Dimension...>) {
    (m_dimensionSizes.push_back(
         detail::compileUniqueComponentCount<Values, Dimension>()),
     ...);
  }

  template <typename Entry>
  static auto runtimeUniqueComponentCount(Entry const &entry,
                                          std::size_t dimension)
      -> std::size_t {
    auto count = std::size_t{};
    auto const &values = entry.values.values();
    for (std::size_t candidate = 0u; candidate < values.size(); ++candidate) {
      auto duplicate = false;
      for (std::size_t previous = 0u; previous < candidate; ++previous)
        if (values.at(previous)[dimension] == values.at(candidate)[dimension])
          duplicate = true;
      if (!duplicate)
        ++count;
    }
    return count;
  }

  template <typename Entry>
  static auto runtimeUniqueComponent(Entry const &entry, std::size_t dimension,
                                     std::size_t uniqueIndex) {
    auto unique = std::size_t{};
    auto const &values = entry.values.values();
    for (std::size_t candidate = 0u; candidate < values.size(); ++candidate) {
      auto duplicate = false;
      for (std::size_t previous = 0u; previous < candidate; ++previous)
        if (values.at(previous)[dimension] == values.at(candidate)[dimension])
          duplicate = true;
      if (!duplicate) {
        if (unique == uniqueIndex)
          return values.at(candidate)[dimension];
        ++unique;
      }
    }
    throw std::out_of_range{
        "A runtime vector component index is out of range."};
  }

  template <typename Entry>
  static auto entrySize(Entry const &entry) -> std::size_t {
    using Values = typename Entry::values_type;
    if constexpr (detail::isRVals<Values>)
      return entry.values.size();
    else
      return Values::size;
  }

  [[nodiscard]] auto rawIndicesFor(std::size_t candidate) const
      -> std::array<std::size_t, dimensionCount> {
    return m_space.indices(candidate);
  }

  [[nodiscard]] auto indicesFor(std::size_t candidate) const
      -> std::array<std::size_t, dimensionCount> {
    return m_space.indices(candidate);
  }

  template <FixedString Name, typename Callable>
  [[nodiscard]] auto
  withCandidateValue(std::array<std::size_t, dimensionCount> const &indices,
                     Callable &&callable) const -> bool {
    using Entry = typename Traits::template entry<Name>;
    constexpr auto entryIndex = Traits::template index<Name>;
    constexpr auto dimensionOffset = Traits::template dimensionOffset<Name>;
    auto const &entry = std::get<entryIndex>(m_tunables.entries());
    using Values = typename Entry::values_type;
    if constexpr (detail::isRVals<Values>) {
      if constexpr (alpaka::isVector_v<typename Values::value_type>) {
        using Vector = typename Values::value_type;
        auto const value = Vector{[&](auto dimension) {
          return runtimeUniqueComponent(entry, dimension,
                                        indices[dimensionOffset + dimension]);
        }};
        return static_cast<bool>(
            std::invoke(std::forward<Callable>(callable), value));
      } else {
        return static_cast<bool>(
            std::invoke(std::forward<Callable>(callable),
                        entry.values.at(indices[dimensionOffset])));
      }
    } else if constexpr (detail::isCTypes<Values> &&
                         detail::candidateDimensionCount<Values> > 1u) {
      return withCompileVectorCandidateValue<Values>(
          indices, dimensionOffset, std::forward<Callable>(callable));
    } else {
      return withCompileCandidateValue<Values>(
          indices[dimensionOffset], std::forward<Callable>(callable));
    }
  }

  template <typename Values, std::size_t Dimension = 0u>
  [[nodiscard]] static auto
  compileVectorComponentAtRuntime(std::size_t selectedIndex,
                                  std::size_t requestedDimension)
      -> decltype(detail::compileVectorComponent<
                  std::tuple_element_t<0u, typename Values::values>, 0u>()) {
    if constexpr (Dimension < detail::candidateDimensionCount<Values>) {
      if (requestedDimension == Dimension)
        return compileUniqueComponentAtRuntime<Values, Dimension>(
            selectedIndex);
      return compileVectorComponentAtRuntime<Values, Dimension + 1u>(
          selectedIndex, requestedDimension);
    } else {
      throw std::out_of_range{
          "A compile-time vector dimension is out of range."};
    }
  }

  template <typename Values, std::size_t Dimension,
            std::size_t UniqueIndex = 0u>
  [[nodiscard]] static auto
  compileUniqueComponentAtRuntime(std::size_t selectedIndex)
      -> decltype(detail::compileVectorComponent<
                  std::tuple_element_t<0u, typename Values::values>,
                  Dimension>()) {
    if constexpr (UniqueIndex <
                  detail::compileUniqueComponentCount<Values, Dimension>()) {
      if (selectedIndex == UniqueIndex)
        return detail::compileUniqueComponent<Values, Dimension, UniqueIndex>();
      return compileUniqueComponentAtRuntime<Values, Dimension,
                                             UniqueIndex + 1u>(selectedIndex);
    } else {
      throw std::out_of_range{
          "A compile-time vector component index is out of range."};
    }
  }

  template <typename Values, typename Callable>
  [[nodiscard]] static auto withCompileVectorCandidateValue(
      std::array<std::size_t, dimensionCount> const &indices,
      std::size_t dimensionOffset, Callable &&callable) -> bool {
    using Prototype = std::tuple_element_t<0u, typename Values::values>;
    using Scalar = decltype(detail::compileVectorComponent<Prototype, 0u>());
    constexpr auto dimensions = detail::candidateDimensionCount<Values>;
    auto const value = alpaka::Vec<Scalar, dimensions>{[&](auto dimension) {
      return compileVectorComponentAtRuntime<Values>(
          indices[dimensionOffset + dimension], dimension);
    }};
    return static_cast<bool>(
        std::invoke(std::forward<Callable>(callable), value));
  }

  template <typename Values, std::size_t ValueIndex = 0u, typename Callable>
  [[nodiscard]] static auto withCompileCandidateValue(std::size_t selectedIndex,
                                                      Callable &&callable)
      -> bool {
    if constexpr (ValueIndex < Values::size) {
      if (selectedIndex == ValueIndex) {
        using Value = std::tuple_element_t<ValueIndex, typename Values::values>;
        if constexpr (detail::isCVals<Values>)
          return static_cast<bool>(
              std::invoke(std::forward<Callable>(callable), Value::value));
        else
          return static_cast<bool>(std::invoke(std::forward<Callable>(callable),
                                               detail::launchValue<Value>()));
      }
      return withCompileCandidateValue<Values, ValueIndex + 1u>(
          selectedIndex, std::forward<Callable>(callable));
    }
    throw std::out_of_range{
        "The compile-time candidate index is outside its tuning dimension."};
  }

  template <typename Restriction>
  [[nodiscard]] auto restrictionAccepted(
      Restriction const &restriction,
      std::array<std::size_t, dimensionCount> const &indices) const -> bool {
    if constexpr (requires { Restriction::second; }) {
      return withCandidateValue<Restriction::first>(
          indices, [this, &restriction, &indices](auto const &firstValue) {
            return withCandidateValue<Restriction::second>(
                indices, [&restriction, &firstValue](auto const &secondValue) {
                  return restriction.accepts(firstValue, secondValue);
                });
          });
    } else {
      return withCandidateValue<Restriction::first>(
          indices, [&restriction](auto const &value) {
            return restriction.accepts(value);
          });
    }
  }

  [[nodiscard]] auto candidateAccepted(
      std::array<std::size_t, dimensionCount> const &indices) const -> bool {
    if constexpr (requires { m_tunables.restrictions(); }) {
      return std::apply(
          [this, &indices](auto const &...restrictions) {
            return (restrictionAccepted(restrictions, indices) && ...);
          },
          m_tunables.restrictions());
    } else {
      return true;
    }
  }

  [[nodiscard]] auto normalizedConfiguration(std::size_t candidate) const
      -> ParameterConfiguration {
    auto const indices = indicesFor(candidate);
    auto configuration = ParameterConfiguration(dimensionCount, 0.0f);
    for (std::size_t dimension = 0u; dimension < dimensionCount; ++dimension) {
      auto const size = m_dimensionSizes[dimension];
      if (size > 1u)
        configuration[dimension] = static_cast<float>(indices[dimension]) /
                                   static_cast<float>(size - 1u);
    }
    return configuration;
  }

  [[nodiscard]] auto
  candidateForConfiguration(ParameterConfiguration const &configuration) const
      -> std::size_t {
    validateParameterConfiguration(configuration, std::span{m_dimensionSizes});
    auto indices = std::array<std::size_t, dimensionCount>{};
    for (std::size_t dimension = 0u; dimension < dimensionCount; ++dimension) {
      auto const size = m_dimensionSizes[dimension];
      auto const coordinate =
          static_cast<long double>(configuration[dimension]);
      indices[dimension] =
          size == 1u ? 0u
                     : static_cast<std::size_t>(std::round(
                           coordinate * static_cast<long double>(size - 1u)));
    }
    auto const candidate = m_space.find(indices);
    if (!candidate)
      throw std::invalid_argument{
          "The configuration is not a registered candidate."};
    return *candidate;
  }

  /** @brief Clamped progress through the adaptive schedule horizon. */
  [[nodiscard]] auto adaptiveProgress() const -> double {
    if (!m_defaults.horizon)
      return 0.0;
    auto const progress = static_cast<double>(m_adaptiveHorizonExecutionCount) /
                          static_cast<double>(*m_defaults.horizon);
    return detail::adaptiveProgressWithActiveHistory(
        progress, m_loadedFromCache, m_defaults.horizonOffsetWithActiveHistory);
  }

  [[nodiscard]] auto candidateUserValid(std::size_t candidate) const noexcept
      -> bool {
    return m_candidateValidity.empty() ||
           m_candidateValidity.at(candidate)->valid;
  }

  [[nodiscard]] auto
  candidateMetricAvailable(std::size_t candidate) const noexcept -> bool {
    return m_missingMetric.empty() || !m_missingMetric.at(candidate);
  }

  [[nodiscard]] auto candidateUsable(std::size_t candidate) const noexcept
      -> bool {
    return candidateUserValid(candidate) && candidateMetricAvailable(candidate);
  }

  [[nodiscard]] auto recordExecution(std::size_t candidate,
                                     std::optional<double> runtimeSeconds)
      -> std::size_t {
    auto const executionIndex = m_executionHistory.size();
    m_executionHistory.push_back(ExecutedConfiguration{
        .executionIndex = executionIndex,
        .candidateIndex = candidate,
        .configuration = normalizedConfiguration(candidate),
        .runtimeSeconds = runtimeSeconds,
        .metricValue = runtimeSeconds,
        .measured = runtimeSeconds.has_value(),
        .valid = ConfigurationValidity{m_candidateValidity.at(candidate)}});
    return executionIndex;
  }

  void removeCandidateFromSchedulers(std::size_t candidate) {
    if (m_budgetDirectCandidate == candidate)
      m_budgetDirectCandidate.reset();
    if (m_queue)
      static_cast<void>(m_queue->retire(candidate));
    if (m_defaults.selection == SelectionPolicy::adaptive &&
        m_scheduled.at(candidate)) {
      m_scheduled.at(candidate) = false;
      --m_scheduledCount;
    }
    if (m_strategy) {
      if (m_space.usesCandidateCatalog())
        m_strategy->candidateInvalidated(candidate,
                                         normalizedConfiguration(candidate));
      else
        m_strategy->configurationInvalidated(
            normalizedConfiguration(candidate));
    }
  }

  void rebuildBestAfterCandidateRemoval() {
    m_bestImprovements.clear();
    m_bestRetiredRuntime = std::numeric_limits<double>::infinity();
    if (auto const best = currentBestCandidate()) {
      m_bestRetiredRuntime = m_histories.at(*best).statistics().estimate();
      if (m_terminal)
        m_bestCandidate = *best;
    } else if (m_terminal) {
      m_bestCandidate = std::numeric_limits<std::size_t>::max();
      m_completionReason = TunerCompletionReason::noValidConfiguration;
    }
  }

  /** @brief Consume application validity decisions before selecting work. */
  auto makeValidityState()
      -> std::shared_ptr<detail::ConfigurationValidityState> {
    auto state = std::make_shared<detail::ConfigurationValidityState>();
    state->generation = m_validityGeneration;
    return state;
  }
  void reconcileUserInvalidations() {
    if (m_consumedValidityGeneration == *m_validityGeneration)
      return;
    m_consumedValidityGeneration = *m_validityGeneration;
    auto changed = false;
    for (std::size_t candidate = 0u; candidate < m_candidateCount;
         ++candidate) {
      auto &state = *m_candidateValidity.at(candidate);
      if (state.valid || m_userInvalidated.at(candidate))
        continue;
      state.invalidationConsumed = true;
      m_userInvalidated.at(candidate) = true;
      ++m_userInvalidatedCount;
      changed = true;

      removeCandidateFromSchedulers(candidate);
    }
    if (!changed)
      return;

    rebuildBestAfterCandidateRemoval();
    if (m_budget)
      refreshBudgetIncumbent();
    writeCache();
  }

  /** @brief Reject a candidate whose preceding online launch has no metric. */
  void reconcileMissingMetric() {
    if constexpr (!usesCustomMetric)
      return;
    if (!m_pendingMetric)
      return;
    auto const pending = *m_pendingMetric;
    m_pendingMetric.reset();
    if (!pending.measure || !candidateUserValid(pending.candidateIndex))
      return;
    if (!m_missingMetric.at(pending.candidateIndex)) {
      m_missingMetric.at(pending.candidateIndex) = true;
      ++m_missingMetricCount;
      removeCandidateFromSchedulers(pending.candidateIndex);
      rebuildBestAfterCandidateRemoval();
      if (m_budget)
        refreshBudgetIncumbent();
    }
    if (!m_budget && m_defaults.selection == SelectionPolicy::fixed &&
        allFixedCandidatesResolved()) {
      finishTuning();
      return;
    }
    writeCache();
  }

  /** @brief Apply the single shared legality and mode-specific admission path.
   *
   * Every recommendation first passes the mandatory tuning-space constraints.
   * Adaptive revisits then pass the optional horizon gates. An enabled queue
   * handles active duplicates only after both admission stages; without one,
   * an accepted recommendation is launched directly.
   */
  [[nodiscard]] auto admitCandidate(std::size_t candidate)
      -> RecommendationDisposition {
    if (!candidateUserValid(candidate)) {
      ++m_userInvalidatedRejectedCount;
      return RecommendationDisposition::userInvalidated;
    }
    if (!candidateMetricAvailable(candidate)) {
      ++m_missingMetricRejectedCount;
      return RecommendationDisposition::metricUnavailable;
    }
    if (m_rejected.at(candidate)) {
      ++m_restrictionRejectedCount;
      return RecommendationDisposition::restrictionRejected;
    }
    if (!candidateAccepted(indicesFor(candidate)) ||
        (m_launchValidator &&
         !m_launchValidator(*this, indicesFor(candidate)))) {
      m_rejected.at(candidate) = true;
      ++m_rejectedCount;
      ++m_restrictionRejectedCount;
      return RecommendationDisposition::restrictionRejected;
    }

    if (m_budget && m_budgetIncumbent && !budgetCandidateWorthwhile(candidate))
      return RecommendationDisposition::budgetRejected;
    auto &history = m_histories.at(candidate);
    auto const revisit = !history.empty();
    if (revisit && m_defaults.selection == SelectionPolicy::fixed &&
        history.isFinished()) {
      ++m_revisitRejectedCount;
      return RecommendationDisposition::revisitRejected;
    }
    if (revisit && m_defaults.selection == SelectionPolicy::adaptive) {
      if (!m_budget && m_defaults.horizon) {
        auto const progress = adaptiveProgress();
        auto const revisitProbability = detail::normalizedLogisticAdmission(
            progress, m_defaults.revisitAdmissionSteepness);
        if (m_admissionDistribution(m_random) >= revisitProbability) {
          ++m_revisitRejectedCount;
          return RecommendationDisposition::revisitRejected;
        }
        auto const best = currentBestCandidate();
        if (!best)
          throw std::logic_error{
              "An adaptive revisit exists without a current best candidate."};
        auto const temperature = detail::adaptiveScoreTemperature(
            progress, m_defaults.scoreTemperatureStart,
            m_defaults.scoreTemperatureEnd);
        auto const scoreProbability = detail::relativeScoreAdmission(
            history.statistics().estimate(),
            m_histories.at(*best).statistics().estimate(), temperature);
        if (m_admissionDistribution(m_random) >= scoreProbability) {
          ++m_scoreRejectedCount;
          return RecommendationDisposition::scoreRejected;
        }
      }
    }
    if (m_queue && m_queue->contains(candidate)) {
      ++m_activeDuplicateAcceptedCount;
      return RecommendationDisposition::activeDuplicate;
    }
    if (revisit && m_defaults.selection == SelectionPolicy::adaptive) {
      history.reopen();
      ++m_revisitAcceptedCount;
    } else if (!revisit) {
      ++m_unseenAcceptedCount;
    }
    if (!m_scheduled.at(candidate)) {
      m_scheduled.at(candidate) = true;
      ++m_scheduledCount;
    }
    return RecommendationDisposition::scheduled;
  }

  /** @brief Translate public configuration into per-candidate history policy.
   */
  [[nodiscard]] auto runtimeHistoryOptions() const
      -> detail::RuntimeHistoryOptions {
    auto const warmupRuns = m_defaults.queue && !m_defaults.queue->disable
                                ? m_defaults.queue->warmupRuns
                                : 0u;
    return {warmupRuns,
            m_defaults.minimumRunsPerCandidate,
            m_defaults.runsPerCandidate,
            m_defaults.ciCheckInterval,
            m_defaults.ciZScore,
            m_defaults.ciRelativeWidth,
            m_defaults.outlierMadScale,
            m_defaults.historyWindowSize,
            m_defaults.selection == SelectionPolicy::fixed};
  }

  /** @brief Whether the online-fixed execution guard has been reached. */
  [[nodiscard]] auto executionBudgetReached() const noexcept -> bool {
    return m_defaults.maximumExecutions &&
           m_executionCount >= *m_defaults.maximumExecutions;
  }

  /** @brief Whether the fixed-mode retired-record guard has been reached. */
  [[nodiscard]] auto retiredConfigurationBudgetReached() const noexcept
      -> bool {
    return m_defaults.maximumRetiredConfigurations &&
           m_retiredConfigurationCount >=
               *m_defaults.maximumRetiredConfigurations;
  }

  [[nodiscard]] auto currentBestCandidate() const
      -> std::optional<std::size_t> {
    auto best = std::optional<std::size_t>{};
    auto bestEstimate = std::numeric_limits<double>::infinity();
    for (std::size_t candidate = 0u; candidate < m_histories.size();
         ++candidate) {
      auto const &history = m_histories.at(candidate);
      if (!candidateUsable(candidate) || m_rejected.at(candidate) ||
          history.empty())
        continue;
      auto const estimate = history.statistics().estimate();
      if (estimate < bestEstimate) {
        bestEstimate = estimate;
        best = candidate;
      }
    }
    return best;
  }

  void retireByMannWhitneyIfWarranted(std::size_t candidate) {
    if (!m_defaults.mannWhitneyEarlyStop)
      return;
    auto &history = m_histories.at(candidate);
    if (history.isFinished())
      return;
    auto const incumbent = currentBestCandidate();
    if (!incumbent || *incumbent == candidate)
      return;
    if (detail::mannWhitneyUCompare(history, m_histories.at(*incumbent),
                                    m_defaults.mannWhitneyMinimumSamples,
                                    m_defaults.mannWhitneyAlpha) ==
        RuntimeComparison::slower)
      history.retire(detail::RuntimeCompletion::mannWhitneyU);
  }

  void recordRetiredConfiguration(std::size_t candidate) {
    ++m_retiredConfigurationCount;
    auto const metricValue = m_histories.at(candidate).statistics().estimate();
    if (metricValue >= m_bestRetiredRuntime)
      return;
    m_bestRetiredRuntime = metricValue;
    m_bestImprovements.push_back(detail::BestImprovement{
        .candidateIndex = candidate,
        .executionCount = m_executionCount,
        .retiredConfigurationCount = m_retiredConfigurationCount,
        .metricValue = metricValue,
        .elapsedSeconds =
            std::chrono::duration<double>{std::chrono::steady_clock::now() -
                                          m_tuningStarted}
                .count()});
  }

  [[nodiscard]] auto runtimeForCandidate(std::size_t candidate) const
      -> std::optional<RuntimeObservation> {
    if (candidate >= m_candidateCount || !candidateUsable(candidate) ||
        m_histories.empty() || m_histories.at(candidate).empty())
      return std::nullopt;
    auto const &history = m_histories.at(candidate);
    auto const statistics = history.statistics();
    auto comparison = RuntimeComparison::unavailable;
    if (auto const incumbent = currentBestCandidate();
        incumbent && *incumbent != candidate)
      comparison = detail::mannWhitneyUCompare(
          history, m_histories.at(*incumbent),
          m_defaults.mannWhitneyMinimumSamples, m_defaults.mannWhitneyAlpha);
    return RuntimeObservation{statistics.estimate(),
                              statistics.sampleCount,
                              statistics.acceptedSampleCount,
                              history.state(),
                              comparison,
                              statistics.confidenceReached};
  }
  [[nodiscard]] auto
  runtimeForConfiguration(ParameterConfiguration const &configuration) const
      -> std::optional<RuntimeObservation> {
    try {
      return runtimeForCandidate(candidateForConfiguration(configuration));
    } catch (std::invalid_argument const &) {
      return std::nullopt;
    }
  }

  class TunerStrategyView final : public StrategyContext {
  public:
    explicit TunerStrategyView(Tuner const &tuner) : m_tuner(tuner) {}

    [[nodiscard]] auto parameterSizes() const noexcept
        -> std::span<std::size_t const> override {
      return m_tuner.m_dimensionSizes;
    }

    [[nodiscard]] auto
    runtimeFor(ParameterConfiguration const &configuration) const
        -> std::optional<RuntimeObservation> override {
      return m_tuner.runtimeForConfiguration(configuration);
    }

    [[nodiscard]] auto hasCandidateCatalog() const noexcept -> bool override {
      return m_tuner.m_space.usesCandidateCatalog();
    }
    [[nodiscard]] auto candidateCount() const noexcept -> std::size_t override {
      return m_tuner.m_candidateCount;
    }
    [[nodiscard]] auto candidateConfiguration(std::size_t id) const
        -> ParameterConfiguration override {
      return m_tuner.normalizedConfiguration(id);
    }
    [[nodiscard]] auto candidateAvailable(std::size_t id) const
        -> bool override {
      return id < m_tuner.m_candidateCount && !m_tuner.m_rejected.at(id) &&
             m_tuner.candidateUsable(id) &&
             (!m_tuner.m_queue || !m_tuner.m_queue->contains(id)) &&
             (m_tuner.m_defaults.selection != SelectionPolicy::fixed ||
              !m_tuner.m_histories.at(id).isFinished());
    }
    [[nodiscard]] auto candidateValid(std::size_t id) const -> bool override {
      return id < m_tuner.m_candidateCount && !m_tuner.m_rejected.at(id) &&
             m_tuner.candidateUsable(id);
    }
    [[nodiscard]] auto candidateObservation(std::size_t id) const
        -> std::optional<RuntimeObservation> override {
      return m_tuner.runtimeForCandidate(id);
    }

  private:
    Tuner const &m_tuner;
  };

  struct CandidateRecommendation {
    std::size_t candidate{};
    RecommendationDisposition disposition{};
  };

  /** @brief Request and process exactly one strategy recommendation. */
  [[nodiscard]] auto recommendCandidate()
      -> std::optional<CandidateRecommendation> {
    if (m_queue && !m_budget &&
        m_defaults.selection == SelectionPolicy::fixed &&
        allFixedCandidatesAdmitted())
      return std::nullopt;
    auto const start = std::chrono::steady_clock::now();
    auto const strategyContext = TunerStrategyView{*this};
    auto recommendation = ParameterConfiguration{};
    auto candidate = std::size_t{};
    if (m_space.usesCandidateCatalog()) {
      auto const selected = m_strategy->recommendCandidate(strategyContext);
      if (!selected)
        return std::nullopt;
      candidate = *selected;
      if (candidate >= m_candidateCount)
        throw std::invalid_argument{
            "Strategy returned an unknown candidate ID."};
      recommendation = normalizedConfiguration(candidate);
    } else {
      recommendation = m_strategy->recommend(strategyContext);
      validateParameterConfiguration(recommendation,
                                     std::span{m_dimensionSizes});
      candidate = candidateForConfiguration(recommendation);
    }
    auto const disposition = admitCandidate(candidate);
    if (m_space.usesCandidateCatalog())
      m_strategy->candidateRecommendationResult(candidate, recommendation,
                                                disposition);
    else
      m_strategy->recommendationResult(recommendation, disposition);
    m_recommendationSecondsSinceLastLaunch +=
        std::chrono::duration<double>{std::chrono::steady_clock::now() - start}
            .count();
    return CandidateRecommendation{candidate, disposition};
  }

  /** @brief Lazily bind persistence, histories, strategy, and optional queue.
   */
  void initialiseScheduling() {
    if (m_schedulingInitialised || m_terminal)
      return;
    m_schedulingInitialised = true;
    m_tuningStarted = std::chrono::steady_clock::now();
    m_startedAtUnixSeconds =
        std::chrono::duration<double>{
            std::chrono::system_clock::now().time_since_epoch()}
            .count();
    m_effectiveRandomSeed =
        detail::contextSeed(m_baseRandomSeed, m_fingerprint);
    m_random.seed(m_effectiveRandomSeed);
    if (m_defaults.strategy == StrategyKind::learnedHybrid)
      m_learnedModelDigest =
          detail::fileFingerprint(m_defaults.learnedModelFile);
    m_histories.assign(m_candidateCount,
                       detail::RuntimeHistory{runtimeHistoryOptions()});
    m_scheduled.assign(m_candidateCount, false);
    m_rejected.assign(m_candidateCount, false);
    m_userInvalidated.assign(m_candidateCount, false);
    m_missingMetric.assign(m_candidateCount, false);
    m_candidateValidity.clear();
    m_candidateValidity.reserve(m_candidateCount);
    for (std::size_t candidate = 0u; candidate < m_candidateCount; ++candidate)
      m_candidateValidity.push_back(makeValidityState());
    m_loadedFromHistoryCandidates.assign(m_candidateCount, false);
    auto const compactLoaded = loadHistory();
    auto const completeLoaded = loadCompleteHistory();
    auto const loaded = compactLoaded || completeLoaded;
    if (loaded && (m_defaults.exploration == ExplorationPolicy::online ||
                   m_defaults.selection == SelectionPolicy::adaptive))
      prepareOnlineRunWithLoadedHistory();
    m_loadedFromCache = loaded || m_loadedLearnedAdapterState.has_value();
    if (m_defaults.exploration == ExplorationPolicy::offline) {
      validateKnownCandidates();
      if (!loaded || !currentBestCandidate())
        throw std::runtime_error{
            "Offline tuning requires a compatible history with at least one "
            "measured configuration."};
      m_explorationComplete = true;
      m_terminal = m_defaults.selection == SelectionPolicy::fixed;
      m_bestCandidate = *currentBestCandidate();
      m_completionReason = TunerCompletionReason::offlineReplay;
      return;
    }
    if (m_space.size() == 0u && m_space.canExpand())
      expandSpace();
    if (m_defaults.strategy == StrategyKind::learnedHybrid) {
      auto const context = learnedModelContext();
      auto const options = LearnedHybridOptions{
          .candidatePoolSize = m_defaults.learnedCandidatePoolSize,
          .candidateBatchSize = m_defaults.learnedCandidateBatchSize};
      m_strategy =
          makeParameterStrategy(m_defaults.strategy, m_effectiveRandomSeed,
                                &context, m_defaults.learnedModelFile, options);
      if (m_loadedLearnedAdapterState || m_loadedCompleteLearnedAdapterState) {
        if (auto *learned =
                dynamic_cast<LearnedHybridStrategy *>(m_strategy.get())) {
          auto restored = false;
          if (m_loadedLearnedAdapterState)
            restored = learned->restoreResidualAdapterState(
                std::move(*m_loadedLearnedAdapterState));
          if (!restored && m_loadedCompleteLearnedAdapterState)
            static_cast<void>(learned->restoreResidualAdapterState(
                std::move(*m_loadedCompleteLearnedAdapterState)));
        }
        m_loadedLearnedAdapterState.reset();
        m_loadedCompleteLearnedAdapterState.reset();
      }
    } else {
      m_strategy =
          makeParameterStrategy(m_defaults.strategy, m_effectiveRandomSeed);
    }
    if (m_space.usesCandidateCatalog() &&
        !m_strategy->supportsCandidateCatalog())
      throw std::invalid_argument{"This custom strategy does not support "
                                  "generated candidate catalogs."};
    if (m_defaults.queue && !m_defaults.queue->disable) {
      m_queue = std::make_unique<detail::CandidateQueue>(
          m_defaults.queue->noiseCancellationWindow,
          m_defaults.queue->maxConsecutiveRuns, false, m_random);
      if (!m_budget)
        refillQueue();
      if (!m_budget && m_terminal && !currentBestCandidate())
        throw std::invalid_argument{
            "The strategy retry limit was reached before any candidate was "
            "accepted and measured."};
      if (!m_budget && !m_terminal && m_queue->empty())
        throw std::invalid_argument{
            "The tuning-space constraints rejected every candidate."};
    }
  }

  /** @brief Start a new online run while retaining one timing history. */
  void prepareOnlineRunWithLoadedHistory() {
    for (auto &history : m_histories)
      history.resetForNewRun();
    m_scheduled.assign(m_candidateCount, false);
    m_rejected.assign(m_candidateCount, false);
    m_loadedFromHistoryCandidates.assign(m_candidateCount, false);
    m_scheduledCount = 0u;
    m_rejectedCount = 0u;
    m_executionCount = 0u;
    m_adaptiveHorizonExecutionCount = 0u;
    m_retiredConfigurationCount = 0u;
    m_unseenAcceptedCount = 0u;
    m_revisitAcceptedCount = 0u;
    m_activeDuplicateAcceptedCount = 0u;
    m_restrictionRejectedCount = 0u;
    m_userInvalidatedRejectedCount = 0u;
    m_missingMetricRejectedCount = 0u;
    m_revisitRejectedCount = 0u;
    m_scoreRejectedCount = 0u;
    m_consecutiveStrategyRetries = 0u;
    m_strategyRetryLimitReachedCount = 0u;
    m_adaptiveRetryFallbackCount = 0u;
    m_bestRetiredRuntime = std::numeric_limits<double>::infinity();
    m_bestImprovements.clear();
    m_bestCandidate = std::numeric_limits<std::size_t>::max();
    m_completionReason = TunerCompletionReason::none;
    m_terminal = false;
    m_explorationComplete = false;
    m_reuseLaunchCount = 0u;
    m_nextProbeCandidate = 0u;
    m_executionBudgetReached = false;
    m_startedAtUnixSeconds =
        std::chrono::duration<double>{
            std::chrono::system_clock::now().time_since_epoch()}
            .count();
  }

  /** @brief Refill vacancies, retrying rejected strategy recommendations.
   *
   * An accepted recommendation resets the retry streak. If active candidates
   * still exist, reaching the limit pauses refill until scheduler progress
   * changes the admission context. With an empty adaptive queue, a measured
   * history incumbent is reopened as a progress fallback; fixed mode instead
   * enters its terminal retry-limit state.
   */
  void refillQueue() {
    applySpaceFeedback();
    if (!m_queue || m_terminal || m_explorationComplete)
      return;
    if (m_defaults.selection == SelectionPolicy::fixed &&
        allFixedCandidatesResolved()) {
      finishTuning();
      return;
    }
    while (!m_queue->full()) {
      if (m_defaults.selection == SelectionPolicy::fixed &&
          allFixedCandidatesAdmitted())
        return;
      if (m_consecutiveStrategyRetries >=
          m_defaults.maximumConsecutiveStrategyRetries) {
        if (m_queue->empty()) {
          if (m_defaults.selection == SelectionPolicy::adaptive) {
            if (auto const fallback = prepareAdaptiveRetryFallback()) {
              if (!m_queue->insert(*fallback))
                throw std::logic_error{
                    "The adaptive retry fallback could not enter the active "
                    "queue."};
              return;
            }
          } else {
            finishTuningAtStrategyRetryLimit();
          }
        }
        return;
      }
      auto const recommendation = recommendCandidate();
      if (!recommendation) {
        ++m_consecutiveStrategyRetries;
        if (m_consecutiveStrategyRetries ==
            m_defaults.maximumConsecutiveStrategyRetries)
          ++m_strategyRetryLimitReachedCount;
        continue;
      }
      if (recommendation->disposition ==
          RecommendationDisposition::activeDuplicate) {
        m_consecutiveStrategyRetries = 0u;
        return;
      }
      if (recommendation->disposition != RecommendationDisposition::scheduled) {
        ++m_consecutiveStrategyRetries;
        if (m_consecutiveStrategyRetries ==
            m_defaults.maximumConsecutiveStrategyRetries)
          ++m_strategyRetryLimitReachedCount;
        continue;
      }
      m_consecutiveStrategyRetries = 0u;
      if (!m_queue->insert(recommendation->candidate))
        throw std::logic_error{
            "An admitted candidate could not enter the active queue."};
      return;
    }
  }

  /** @brief Reopen the measured incumbent without ending adaptive tuning.
   *
   * This bounds synchronous strategy work when every proposal in one refill
   * attempt is rejected. The fallback remains a normal measured queue
   * activation, advances the adaptive horizon, and leaves the strategy active
   * for the next refill.
   */
  [[nodiscard]] auto prepareAdaptiveRetryFallback()
      -> std::optional<std::size_t> {
    auto const best = currentBestCandidate();
    if (!best)
      return std::nullopt;
    if (m_scheduled.at(*best) || m_rejected.at(*best) ||
        !candidateUsable(*best))
      throw std::logic_error{"The adaptive retry fallback is not schedulable."};
    m_histories.at(*best).reopen();
    m_scheduled.at(*best) = true;
    ++m_scheduledCount;
    m_consecutiveStrategyRetries = 0u;
    ++m_adaptiveRetryFallbackCount;
    return best;
  }

  /** @brief Whether every legal fixed-mode history has retired. */
  [[nodiscard]] auto allFixedCandidatesAdmitted() -> bool {
    replenishResolvedSpace();
    for (std::size_t candidate = 0u; candidate < m_candidateCount; ++candidate)
      if (!m_scheduled.at(candidate) && !m_rejected.at(candidate) &&
          candidateUsable(candidate))
        return false;
    return true;
  }

  /** @brief Whether every valid fixed-mode history has retired. */
  [[nodiscard]] auto allFixedCandidatesResolved() -> bool {
    replenishResolvedSpace();
    for (std::size_t candidate = 0u; candidate < m_candidateCount;
         ++candidate) {
      if (!m_rejected.at(candidate) && candidateUsable(candidate) &&
          !m_histories.at(candidate).isFinished())
        return false;
    }
    return true;
  }

  /** @brief Obtain one accepted strategy result without queue residency. */
  [[nodiscard]] auto nextDirectCandidate()
      -> detail::CandidateQueue::Selection {
    while (!m_explorationComplete) {
      applySpaceFeedback();
      if (m_explorationComplete)
        break;
      if (m_defaults.selection == SelectionPolicy::fixed &&
          allFixedCandidatesResolved()) {
        finishTuning();
        break;
      }
      if (m_consecutiveStrategyRetries >=
          m_defaults.maximumConsecutiveStrategyRetries) {
        if (m_defaults.selection == SelectionPolicy::adaptive) {
          if (auto const fallback = prepareAdaptiveRetryFallback())
            return {*fallback, true, true, true};
        } else {
          finishTuningAtStrategyRetryLimit();
        }
        break;
      }
      auto const recommendation = recommendCandidate();
      if (recommendation &&
          recommendation->disposition == RecommendationDisposition::scheduled) {
        m_consecutiveStrategyRetries = 0u;
        return {recommendation->candidate, true, true, true};
      }
      ++m_consecutiveStrategyRetries;
      if (m_consecutiveStrategyRetries ==
          m_defaults.maximumConsecutiveStrategyRetries)
        ++m_strategyRetryLimitReachedCount;
    }
    if (m_bestCandidate == std::numeric_limits<std::size_t>::max())
      throw std::invalid_argument{
          "No valid measured configuration is available to launch."};
    if (!m_terminal)
      return nextKnownCandidate();
    return {m_bestCandidate, false, false, false};
  }

  void validateKnownCandidates() {
    for (std::size_t candidate = 0u; candidate < m_candidateCount;
         ++candidate) {
      if (m_histories.at(candidate).empty() || !candidateUsable(candidate) ||
          m_rejected.at(candidate))
        continue;
      if (!candidateAccepted(indicesFor(candidate)) ||
          (m_launchValidator &&
           !m_launchValidator(*this, indicesFor(candidate)))) {
        m_rejected.at(candidate) = true;
        ++m_rejectedCount;
      }
    }
  }

  /** @brief Select the incumbent or a periodic measured known alternative. */
  [[nodiscard]] auto nextKnownCandidate() -> detail::CandidateQueue::Selection {
    validateKnownCandidates();
    auto const best = currentBestCandidate();
    if (!best) {
      m_terminal = true;
      m_completionReason = TunerCompletionReason::noValidConfiguration;
      writeCache();
      throw std::logic_error{
          "No valid measured configuration is available to launch."};
    }
    auto candidate = *best;
    if ((m_reuseLaunchCount + 1u) % m_defaults.adaptiveProbeInterval == 0u) {
      for (std::size_t offset = 0u; offset < m_candidateCount; ++offset) {
        auto const alternative =
            (m_nextProbeCandidate + offset) % m_candidateCount;
        if (alternative != *best && candidateUsable(alternative) &&
            !m_rejected.at(alternative) &&
            !m_histories.at(alternative).empty() &&
            candidateAccepted(indicesFor(alternative)) &&
            (!m_launchValidator ||
             m_launchValidator(*this, indicesFor(alternative)))) {
          candidate = alternative;
          break;
        }
      }
    }
    return {candidate, true, true, true};
  }

  /** @brief Select an admitted candidate or a terminal winner replay. */
  [[nodiscard]] auto nextCandidate() -> detail::CandidateQueue::Selection {
    if (m_explorationComplete && !m_terminal)
      return nextKnownCandidate();
    if (!m_queue)
      return nextDirectCandidate();
    refillQueue();
    if (m_explorationComplete && !m_terminal)
      return nextKnownCandidate();
    if (m_terminal) {
      if (m_bestCandidate == std::numeric_limits<std::size_t>::max())
        throw std::logic_error{
            "No valid measured configuration is available to replay."};
      return detail::CandidateQueue::Selection{m_bestCandidate, false, false,
                                               false};
    }
    auto const selected = m_queue->next();
    if (selected)
      return *selected;
    throw std::logic_error{"The tuning scheduler has no admitted candidate."};
  }

  template <typename Queue, typename FrameSpec, typename Bundle>
  struct LaunchCall {
    Tuner *tuner;
    Queue const *queue;
    FrameSpec const *frameSpec;
    Bundle const *prototype;
    std::optional<double> runtimeSeconds;
  };

  struct PendingMetric {
    std::size_t executionIndex{};
    std::size_t candidateIndex{};
    bool measure{};
    bool beginActivation{};
    bool endActivation{};
  };

  using CompileVariantFunctor =
      std::function<void(void *, std::size_t, bool, bool, bool, bool)>;

  [[nodiscard]] auto compileVariantKey(
      std::array<std::size_t, dimensionCount> const &indices) const
      -> std::string {
    std::ostringstream key;
    appendCompileVariantKey<0u, 0u>(key, indices);
    return key.str();
  }

  template <std::size_t EntryIndex, std::size_t DimensionOffset>
  static void appendCompileVariantKey(
      std::ostringstream &key,
      std::array<std::size_t, dimensionCount> const &indices) {
    if constexpr (EntryIndex < std::tuple_size_v<Entries>) {
      using Entry = std::tuple_element_t<EntryIndex, Entries>;
      using Values = typename Entry::values_type;
      constexpr auto dimensions = detail::candidateDimensionCount<Values>;
      if constexpr (detail::isCompileTimeCandidates<Values>) {
        for (std::size_t dimension = 0u; dimension < dimensions; ++dimension)
          key << Entry::nameView() << '[' << dimension
              << "]=" << indices[DimensionOffset + dimension] << ';';
      }
      appendCompileVariantKey<EntryIndex + 1u, DimensionOffset + dimensions>(
          key, indices);
    }
  }

  template <typename Queue, typename FrameSpec, typename Bundle>
  void initialiseCompileVariants() {
    auto const signature =
        detail::typeName<Bundle>() + "|" + detail::typeName<Queue>();
    if (!m_compileVariants.empty()) {
      if (m_compileVariantSignature == signature)
        return;
      if (!m_defaults.replayFastPath || !m_terminal)
        throw std::logic_error{
            "A tuner is bound to one KernelBundle and queue type."};
      m_compileVariants.clear();
      m_compileValidators.clear();
      m_compileVariantIndices.clear();
    }
    std::array<std::size_t, dimensionCount> indices{};
    appendCompileVariants<Queue, FrameSpec, Bundle, 0u, 0u>(indices);
    m_compileVariantSignature = signature;
  }

  template <typename Queue, typename FrameSpec, typename Bundle,
            std::size_t EntryIndex, std::size_t DimensionOffset,
            typename... CompileValues>
  void appendCompileVariants(std::array<std::size_t, dimensionCount> &indices) {
    if constexpr (EntryIndex == std::tuple_size_v<Entries>) {
      auto const key = compileVariantKey(indices);
      m_compileVariantIndices.emplace(key, m_compileVariants.size());
      m_compileValidators.emplace_back(
          [](Tuner const &self, void const *rawLaunch, void const *rawBundle,
             CandidateIndices const &tuple) {
            auto const &launch = *static_cast<FrameSpec const *>(rawLaunch);
            auto const &prototype = *static_cast<Bundle const *>(rawBundle);
            auto const bundle =
                self.template rebuildBundle<CompileValues...>(prototype, tuple);
            return self.template withLaunchSpec<CompileValues...>(
                launch, bundle, tuple, [&](auto const &spec) {
                  return self.hardwareLaunchLegal(spec, bundle);
                });
          });
      m_compileVariants.emplace_back(
          [](void *rawCall, std::size_t candidate, bool measure,
             bool beginActivation, bool endActivation, bool trackExecution) {
            auto &call =
                *static_cast<LaunchCall<Queue, FrameSpec, Bundle> *>(rawCall);
            call.tuner->template launchCandidate<CompileValues...>(
                *call.queue, *call.frameSpec, *call.prototype, candidate,
                measure, beginActivation, endActivation, trackExecution,
                call.runtimeSeconds);
          });
    } else {
      using Entry = std::tuple_element_t<EntryIndex, Entries>;
      using Values = typename Entry::values_type;
      constexpr auto dimensions = detail::candidateDimensionCount<Values>;
      if constexpr (detail::isRVals<Values>) {
        appendCompileVariants<Queue, FrameSpec, Bundle, EntryIndex + 1u,
                              DimensionOffset + dimensions, CompileValues...>(
            indices);
      } else if constexpr (detail::isCTypes<Values> && dimensions > 1u) {
        appendCompileVectorVariants<Queue, FrameSpec, Bundle, EntryIndex,
                                    DimensionOffset, Values, 0u, 0u,
                                    std::index_sequence<>, CompileValues...>(
            indices);
      } else {
        appendCValVariants<Queue, FrameSpec, Bundle, EntryIndex,
                           DimensionOffset, Values, 0u, CompileValues...>(
            indices);
      }
    }
  }

  template <typename Queue, typename FrameSpec, typename Bundle,
            std::size_t EntryIndex, std::size_t DimensionOffset,
            typename Values, std::size_t ValueIndex, typename... CompileValues>
  void appendCValVariants(std::array<std::size_t, dimensionCount> &indices) {
    if constexpr (ValueIndex < Values::size) {
      using Entry = std::tuple_element_t<EntryIndex, Entries>;
      using Value = std::tuple_element_t<ValueIndex, typename Values::values>;
      indices[DimensionOffset] = ValueIndex;
      appendCompileVariants<Queue, FrameSpec, Bundle, EntryIndex + 1u,
                            DimensionOffset + 1u, CompileValues...,
                            detail::SelectedCompileValue<Entry::name, Value>>(
          indices);
      appendCValVariants<Queue, FrameSpec, Bundle, EntryIndex, DimensionOffset,
                         Values, ValueIndex + 1u, CompileValues...>(indices);
    }
  }

  template <typename Queue, typename FrameSpec, typename Bundle,
            std::size_t EntryIndex, std::size_t DimensionOffset,
            typename Values, std::size_t VectorDimension,
            std::size_t ValueIndex, typename SelectedIndices,
            typename... CompileValues>
  void appendCompileVectorVariants(
      std::array<std::size_t, dimensionCount> &indices) {
    constexpr auto dimensions = detail::candidateDimensionCount<Values>;
    if constexpr (VectorDimension == dimensions) {
      using Entry = std::tuple_element_t<EntryIndex, Entries>;
      using Prototype = std::tuple_element_t<0u, typename Values::values>;
      using Value =
          detail::reconstructCompileVector<Values, Prototype, SelectedIndices>;
      appendCompileVariants<Queue, FrameSpec, Bundle, EntryIndex + 1u,
                            DimensionOffset + dimensions, CompileValues...,
                            detail::SelectedCompileValue<Entry::name, Value>>(
          indices);
    } else if constexpr (ValueIndex < detail::compileUniqueComponentCount<
                                          Values, VectorDimension>()) {
      indices[DimensionOffset + VectorDimension] = ValueIndex;
      using NextIndices =
          detail::appendIndexSequence<SelectedIndices, ValueIndex>;
      appendCompileVectorVariants<Queue, FrameSpec, Bundle, EntryIndex,
                                  DimensionOffset, Values, VectorDimension + 1u,
                                  0u, NextIndices, CompileValues...>(indices);
      appendCompileVectorVariants<
          Queue, FrameSpec, Bundle, EntryIndex, DimensionOffset, Values,
          VectorDimension, ValueIndex + 1u, SelectedIndices, CompileValues...>(
          indices);
    }
  }

  template <FixedString Name, typename... CompileValues>
  [[nodiscard]] auto
  valueFor(std::array<std::size_t, dimensionCount> const &indices) const {
    using Entry = typename Traits::template entry<Name>;
    constexpr auto entryIndex = Traits::template index<Name>;
    constexpr auto dimensionOffset = Traits::template dimensionOffset<Name>;
    auto const &entry = std::get<entryIndex>(m_tunables.entries());
    using Values = typename Entry::values_type;
    if constexpr (detail::isRVals<Values>) {
      if constexpr (alpaka::isVector_v<typename Values::value_type>) {
        using Vector = typename Values::value_type;
        return Vector{[&](auto dimension) {
          return runtimeUniqueComponent(entry, dimension,
                                        indices[dimensionOffset + dimension]);
        }};
      } else {
        return entry.values.at(indices[dimensionOffset]);
      }
    } else {
      using Selected =
          typename detail::FindSelectedCompile<Name, CompileValues...>::type;
      return Selected{};
    }
  }

  template <FixedString Name, typename... CompileValues>
  [[nodiscard]] auto
  launchValueFor(std::array<std::size_t, dimensionCount> const &indices) const {
    using Entry = typename Traits::template entry<Name>;
    using Values = typename Entry::values_type;
    if constexpr (detail::isRVals<Values>) {
      return valueFor<Name, CompileValues...>(indices);
    } else {
      using Selected =
          typename detail::FindSelectedCompile<Name, CompileValues...>::type;
      if constexpr (detail::isCVals<Values>)
        return Selected::value;
      else
        return detail::launchValue<Selected>();
    }
  }

  template <typename Argument, typename... CompileValues>
  [[nodiscard]] auto rebuildArgument(
      Argument const &argument,
      std::array<std::size_t, dimensionCount> const &indices) const {
    if constexpr (detail::isMarkedTunable<Argument>)
      return valueFor<Argument::name, CompileValues...>(indices);
    else
      return argument;
  }

  template <typename... CompileValues, typename Bundle>
  [[nodiscard]] auto
  rebuildBundle(Bundle const &prototype,
                std::array<std::size_t, dimensionCount> const &indices) const {
    return alpaka::apply(
        [this, &prototype, &indices](auto const &...arguments) {
          return alpaka::KernelBundle{
              prototype.getKernelFn(),
              rebuildArgument<std::remove_cvref_t<decltype(arguments)>,
                              CompileValues...>(arguments, indices)...};
        },
        prototype.getArgs());
  }

  template <typename... CompileValues, typename LaunchSpec, typename Bundle,
            typename Callable>
  decltype(auto)
  withLaunchSpec(LaunchSpec const &prototypeLaunch, Bundle const &bundle,
                 CandidateIndices const &indices, Callable &&callable) const {

    if constexpr (alpaka::onHost::isFrameSpec_v<
                      std::remove_cvref_t<LaunchSpec>>) {
      auto const numFrames = [&] {
        if constexpr (Traits::template has<detail::numFramesName>)
          return launchValueFor<detail::numFramesName, CompileValues...>(
              indices);
        else
          return prototypeLaunch.getNumFrames();
      }();
      auto const frameExtent = [&] {
        if constexpr (Traits::template has<detail::frameExtentName>)
          return launchValueFor<detail::frameExtentName, CompileValues...>(
              indices);
        else
          return prototypeLaunch.getFrameExtents();
      }();
      auto const frame = alpaka::onHost::FrameSpec{
          numFrames, frameExtent,
          std::remove_cvref_t<LaunchSpec>::getExecutor()};
      if constexpr (Traits::template has<detail::numThreadsName> ||
                    Traits::template has<detail::numBlocksName>) {
        auto const derived = deriveThreadSpec(m_device, frame, bundle);
        auto const blocks = [&] {
          if constexpr (Traits::template has<detail::numBlocksName>)
            return launchValueFor<detail::numBlocksName, CompileValues...>(
                indices);
          else
            return derived.getNumBlocks();
        }();
        auto const threads = [&] {
          if constexpr (Traits::template has<detail::numThreadsName>)
            return launchValueFor<detail::numThreadsName, CompileValues...>(
                indices);
          else
            return derived.getNumThreads();
        }();
        auto const thread = alpaka::onHost::ThreadSpec{
            blocks, threads,
            std::remove_cvref_t<decltype(derived)>::getExecutor()};
        return std::invoke(callable, thread);
      } else {
        return std::invoke(callable, frame);
      }
    } else if constexpr (alpaka::onHost::isThreadSpec_v<
                             std::remove_cvref_t<LaunchSpec>>) {
      auto const blocks = [&] {
        if constexpr (Traits::template has<detail::numBlocksName>)
          return launchValueFor<detail::numBlocksName, CompileValues...>(
              indices);
        else
          return prototypeLaunch.getNumBlocks();
      }();
      auto const threads = [&] {
        if constexpr (Traits::template has<detail::numThreadsName>)
          return launchValueFor<detail::numThreadsName, CompileValues...>(
              indices);
        else
          return prototypeLaunch.getNumThreads();
      }();
      auto const thread = alpaka::onHost::ThreadSpec{
          blocks, threads, std::remove_cvref_t<LaunchSpec>::getExecutor()};
      return std::invoke(callable, thread);
    } else {
      static_assert(alpaka::onHost::concepts::ThreadOrFrameSpec<
                        std::remove_cvref_t<LaunchSpec>>,
                    "alpakaTune::Tuner::enqueue requires "
                    "alpaka::onHost::FrameSpec or ThreadSpec.");
    }
  }

  template <typename Spec, typename Bundle>
  [[nodiscard]] auto hardwareLaunchLegal(Spec const &spec,
                                         Bundle const &bundle) const -> bool {
    if constexpr (alpaka::onHost::isFrameSpec_v<Spec>) {
      auto const extents = spec.getFrameExtents();
      auto const frames = spec.getNumFrames();
      std::size_t elements{1u};
      for (std::size_t d{}; d < ALPAKA_TYPEOF(extents)::dim(); ++d) {
        if (extents[d] <= 0 || frames[d] <= 0 ||
            static_cast<std::size_t>(extents[d]) >
                std::numeric_limits<std::size_t>::max() / elements)
          return false;
        elements *= static_cast<std::size_t>(extents[d]);
      }
      return hardwareLaunchLegal(deriveThreadSpec(m_device, spec, bundle),
                                 bundle);
    } else {
      auto const properties = m_device.getDeviceProperties();
      auto const threads = spec.getNumThreads();
      auto const blocks = spec.getNumBlocks();
      constexpr auto dimensions = ALPAKA_TYPEOF(threads)::dim();
      auto const threadLimits =
          properties.template getMaxThreadsPerBlock<dimensions>();
      auto const blockLimits =
          properties.template getMaxBlocksPerGrid<dimensions>();
      std::size_t threadProduct{1u}, blockProduct{1u};
      for (std::size_t d{}; d < dimensions; ++d) {
        if (threads[d] <= 0 || blocks[d] <= 0 || threads[d] > threadLimits[d] ||
            blocks[d] > blockLimits[d])
          return false;
        if (static_cast<std::size_t>(threads[d]) >
                properties.maxThreadsPerBlock / threadProduct ||
            static_cast<std::size_t>(blocks[d]) >
                properties.maxBlocksPerGrid / blockProduct)
          return false;
        threadProduct *= static_cast<std::size_t>(threads[d]);
        blockProduct *= static_cast<std::size_t>(blocks[d]);
      }
      if constexpr (alpaka::isSeqExecutor(Spec::getExecutor()))
        if (threadProduct != 1u)
          return false;
      return alpaka::onHost::getDynSharedMemBytes(spec, bundle) <=
             properties.sharedMemPerBlockBytes;
    }
  }

  template <typename Queue, typename LaunchSpec, typename Bundle>
  void bindLaunchValidator(LaunchSpec const &launch, Bundle const &bundle) {
    if constexpr (Traits::generatesCandidates) {
      initialiseCompileVariants<Queue, LaunchSpec, Bundle>();
      m_launchValidator = [launch, bundle](Tuner const &self,
                                           CandidateIndices const &indices) {
        return self.m_compileValidators.at(self.m_compileVariantIndices.at(
            self.compileVariantKey(indices)))(self, &launch, &bundle, indices);
      };
    }
  }

  template <typename... CompileValues, typename Queue, typename LaunchSpec,
            typename Bundle>
  void launchCandidate(Queue const &queue, LaunchSpec const &prototypeLaunch,
                       Bundle const &prototype, std::size_t candidate,
                       bool measure, bool beginActivation, bool endActivation,
                       bool trackExecution,
                       std::optional<double> &runtimeSeconds) {
    auto const indices = indicesFor(candidate);
    auto const bundle = rebuildBundle<CompileValues...>(prototype, indices);

    auto launch = [&](auto const &launchQueue) {
      withLaunchSpec<CompileValues...>(
          prototypeLaunch, bundle, indices, [&](auto const &spec) {
            if constexpr (Traits::generatesCandidates)
              if (!hardwareLaunchLegal(spec, bundle))
                throw std::invalid_argument{
                    "The selected launch geometry exceeds current kernel or "
                    "device resources."};
            launchQueue.enqueue(spec, bundle);
          });
    };

    if (trackExecution) {
      ++m_executionCount;
      if (!m_explorationComplete &&
          m_defaults.selection == SelectionPolicy::adaptive)
        ++m_adaptiveHorizonExecutionCount;
    }
    if (!measure) {
      launch(queue);
      return;
    }
    if constexpr (usesCustomMetric) {
      launch(queue);
    } else {
      if constexpr (std::same_as<ALPAKA_TYPEOF(queue.getTiming()),
                                 alpaka::timing::Enabled>) {
        if (!m_measurementTimer.timer)
          m_measurementTimer.timer.emplace(m_device);
        runtimeSeconds = m_measurementTimer.timer->measure(queue, launch);
      } else {
        throw std::invalid_argument{
            "A measured tuning launch requires a timing-enabled queue."};
      }
      recordInstrumentationOverheadWarning(*runtimeSeconds);
      recordMetric(candidate, *runtimeSeconds, beginActivation, endActivation);
    }
  }

  /** @brief Incorporate one timing or application-provided objective value. */
  void recordMetric(std::size_t candidate, double value, bool beginActivation,
                    bool endActivation) {
    if (m_budget) {
      recordBudgetMetric(candidate, value);
      return;
    }
    auto &history = m_histories.at(candidate);
    if (m_explorationComplete &&
        m_defaults.selection == SelectionPolicy::adaptive) {
      history.reopen();
      history.beginActivation(false);
      static_cast<void>(history.record(value));
      history.retire(detail::RuntimeCompletion::adaptiveVisit);
      stageCache(candidate, true);
      return;
    }
    if (beginActivation)
      history.beginActivation();
    static_cast<void>(history.record(value));
    if (m_defaults.selection == SelectionPolicy::fixed)
      retireByMannWhitneyIfWarranted(candidate);
    if (m_defaults.selection == SelectionPolicy::adaptive && endActivation)
      history.retire(detail::RuntimeCompletion::adaptiveVisit);
    auto const schedulingChanged = history.isFinished();
    auto const completedScore = history.statistics().estimate();
    if (schedulingChanged) {
      recordRetiredConfiguration(candidate);
      m_refinementPending =
          m_space.observe(candidate, completedScore) || m_refinementPending;
      if (m_queue && !m_queue->retire(candidate))
        throw std::logic_error{
            "The completed configuration was not active in the queue."};
      if (m_defaults.selection == SelectionPolicy::adaptive) {
        m_scheduled.at(candidate) = false;
        --m_scheduledCount;
      }
    }
    if (!m_explorationComplete && executionBudgetReached()) {
      finishTuningAtBudget(TunerCompletionReason::maximumExecutions);
      return;
    }
    if (!m_explorationComplete && retiredConfigurationBudgetReached()) {
      finishTuningAtBudget(TunerCompletionReason::maximumRetiredConfigurations);
      return;
    }
    if (schedulingChanged && finishResolvedCatalog())
      return;
    if (schedulingChanged && m_queue) {
      m_consecutiveStrategyRetries = 0u;
      refillQueue();
    }
    if (m_defaults.selection == SelectionPolicy::fixed && !m_queue &&
        allFixedCandidatesResolved()) {
      finishTuning();
      return;
    }
    stageCache(candidate, schedulingChanged);
  }

  void recordInstrumentationOverheadWarning(double runtimeSeconds) {
    constexpr auto thresholdSeconds = 200.0e-6;
    if (m_instrumentationOverheadWarning || runtimeSeconds >= thresholdSeconds)
      return;
    m_instrumentationOverheadWarning = InstrumentationOverheadWarning{
        .observedRuntimeSeconds = runtimeSeconds};
    std::clog
        << "alpakaTune warning: measured runtime for " << m_kernelName << " ("
        << runtimeSeconds * 1.0e6
        << " us) is below 200 us; per-launch instrumentation and "
           "synchronization commonly costs approximately 20-40 us and may "
           "outweigh tuning gains. This warning is emitted once for this "
           "tuner context.\n";
  }

  template <typename LaunchSpec, typename Bundle>
  [[nodiscard]] auto makeFingerprint(LaunchSpec const &launchSpec,
                                     Bundle const &prototype) const
      -> std::string {
    using BundleType = std::remove_cvref_t<Bundle>;
    std::ostringstream identity;
    identity << "schema=" << completeHistorySchemaVersion << '\n';
    identity << "kernel=" << detail::typeName<typename BundleType::KernelFn>()
             << '\n';
    identity << "bundle=" << detail::typeName<BundleType>() << '\n';
    identity << "device=" << detail::identityName(m_device) << '\n';
    identity << "metric=" << tuningMetricKindName(metricKind()) << ':'
             << m_metricName << '\n';
    identity << "executor="
             << detail::printable(
                    std::remove_cvref_t<LaunchSpec>::getExecutor())
             << '\n';
    if constexpr (alpaka::onHost::concepts::FrameSpec<
                      std::remove_cvref_t<LaunchSpec>>) {
      identity << "frame=" << detail::printable(launchSpec.getNumFrames())
               << ':' << detail::printable(launchSpec.getFrameExtents())
               << '\n';
    } else if constexpr (alpaka::onHost::concepts::ThreadSpec<
                             std::remove_cvref_t<LaunchSpec>>) {
      identity << "thread=" << detail::printable(launchSpec.getNumBlocks())
               << ':' << detail::printable(launchSpec.getNumThreads()) << '\n';
    } else {
      static_assert(alpaka::onHost::concepts::ThreadOrFrameSpec<
                        std::remove_cvref_t<LaunchSpec>>,
                    "alpakaTune::Tuner::enqueue requires "
                    "alpaka::onHost::FrameSpec or ThreadSpec.");
    }
    for (auto const &entry : m_identityEntries)
      identity << "context=" << entry << '\n';
    std::apply(
        [&identity](auto const &...entries) {
          (appendTunableFingerprint(identity, entries), ...);
        },
        m_tunables.entries());
    if (!m_space.usesCandidateCatalog())
      identity << "candidates=" << m_candidateCount << '\n';
    else
      identity << "space-generator=1\n";
    if constexpr (requires { m_tunables.restrictions(); }) {
      std::apply(
          [&identity](auto const &...restrictions) {
            (appendRestrictionFingerprint(identity, restrictions), ...);
          },
          m_tunables.restrictions());
    }
    static_cast<void>(prototype);
    return detail::hashFingerprint(identity.str());
  }

  template <typename Entry>
  static void appendTunableFingerprint(std::ostringstream &identity,
                                       Entry const &entry) {
    using Values = typename Entry::values_type;
    identity << entry.nameView() << '=';
    if constexpr (requires {
                    entry.values.minimum();
                    entry.values.maximum();
                    entry.values.step();
                  }) {
      identity << "interval:" << detail::printable(entry.values.minimum())
               << ':' << detail::printable(entry.values.maximum()) << ':'
               << detail::printable(entry.values.step());
    } else if constexpr (detail::isRVals<Values>) {
      for (auto const &value : entry.values.values())
        identity << detail::printable(value) << ',';
    } else {
      appendCValFingerprint<Values>(identity);
    }
    if constexpr (detail::isAutomaticCandidates<Values>)
      identity << ":auto:" << entry.values.logarithmic() << ':'
               << entry.values.categorical() << ':' << entry.values.alignment()
               << ":preferred="
               << entry.values.preferredIndex().value_or(
                      std::numeric_limits<std::size_t>::max());
    if constexpr (requires { entry.values.generator; })
      identity << ":dependent="
               << detail::typeName<decltype(entry.values.generator)>();
    identity << ';';
  }

  /** Stable workload key excluding strategy, seed, budgets, and device. */
  [[nodiscard]] auto modelWorkloadId() const -> std::string {
    std::ostringstream identity;
    identity << "model-feature-schema="
             << (m_space.usesCandidateCatalog() ? 2u : 1u) << '\n';
    identity << "kernel=" << m_kernelName << '\n';
    identity << "launch=" << m_launchSpecification << '\n';
    identity << "metric=" << tuningMetricKindName(metricKind()) << ':'
             << m_metricName << '\n';
    for (auto const &entry : m_identityEntries)
      identity << "context=" << entry << '\n';
    std::apply(
        [&identity](auto const &...entries) {
          (appendTunableFingerprint(identity, entries), ...);
        },
        m_tunables.entries());
    if (!m_space.usesCandidateCatalog())
      identity << "candidates=" << m_candidateCount << '\n';
    else
      identity << "space-generator=1\n";
    return detail::hashFingerprint(identity.str());
  }

  template <typename Restriction>
  static void appendRestrictionFingerprint(std::ostringstream &identity,
                                           Restriction const &) {
    using Type = std::remove_cvref_t<Restriction>;
    identity << "restriction=" << Type::first.view();
    if constexpr (requires { Type::second; })
      identity << ':' << Type::second.view();
    identity << ':' << detail::typeName<Type>() << '\n';
  }

  template <typename Values, std::size_t Index = 0u>
  static void appendCValFingerprint(std::ostringstream &identity) {
    if constexpr (Index < Values::size) {
      using Value = std::tuple_element_t<Index, typename Values::values>;
      if constexpr (detail::isCVals<Values>)
        identity << detail::printable(Value::value) << ',';
      else
        identity << detail::printable(Value{}) << ',';
      appendCValFingerprint<Values, Index + 1u>(identity);
    }
  }

  template <typename LaunchSpec>
  [[nodiscard]] static auto launchDescription(LaunchSpec const &launchSpec)
      -> std::string {
    std::ostringstream description;
    if constexpr (alpaka::onHost::concepts::FrameSpec<
                      std::remove_cvref_t<LaunchSpec>>)
      description << "FrameSpec{"
                  << detail::printable(launchSpec.getNumFrames()) << ','
                  << detail::printable(launchSpec.getFrameExtents())
                  << ", executor="
                  << detail::printable(
                         std::remove_cvref_t<LaunchSpec>::getExecutor())
                  << '}';
    else if constexpr (alpaka::onHost::concepts::ThreadSpec<
                           std::remove_cvref_t<LaunchSpec>>)
      description << "ThreadSpec{"
                  << detail::printable(launchSpec.getNumBlocks()) << ','
                  << detail::printable(launchSpec.getNumThreads())
                  << ", executor="
                  << detail::printable(
                         std::remove_cvref_t<LaunchSpec>::getExecutor())
                  << '}';
    return description.str();
  }

  void bindFingerprint(std::string const &fingerprint, std::string kernelName,
                       std::string launchSpecification) {
    if (m_fingerprint.empty()) {
      m_fingerprint = fingerprint;
      m_kernelName = std::move(kernelName);
      m_launchSpecification = std::move(launchSpecification);
    } else if (m_fingerprint != fingerprint) {
      throw std::logic_error{
          "A tuner is bound to one kernel and one launch configuration."};
    }
  }

  [[nodiscard]] static auto completionReasonFromName(std::string_view name)
      -> TunerCompletionReason {
    if (name == "insufficient_expected_benefit")
      return TunerCompletionReason::insufficientExpectedBenefit;
    if (name == "tuning_overhead_budget")
      return TunerCompletionReason::tuningOverheadBudget;
    if (name == "all_configurations")
      return TunerCompletionReason::allConfigurations;
    if (name == "offline_replay")
      return TunerCompletionReason::offlineReplay;
    if (name == "maximum_executions")
      return TunerCompletionReason::maximumExecutions;
    if (name == "maximum_retired_configurations")
      return TunerCompletionReason::maximumRetiredConfigurations;
    if (name == "maximum_consecutive_strategy_retries")
      return TunerCompletionReason::maximumConsecutiveStrategyRetries;
    if (name == "candidate_budget")
      return TunerCompletionReason::candidateBudget;
    if (name == "plateau")
      return TunerCompletionReason::plateau;
    if (name == "generation_stalled")
      return TunerCompletionReason::generationStalled;
    if (name == "no_valid_configuration")
      return TunerCompletionReason::noValidConfiguration;
    return TunerCompletionReason::none;
  }

#if ALPAKA_TUNE_HAS_JSON
  [[nodiscard]] auto serializedCandidateSpace() const -> nlohmann::json {
    auto const info = m_space.info();
    return {{"declared_combination_count",
             info.declaredCombinationCount
                 ? nlohmann::json(*info.declaredCombinationCount)
                 : nlohmann::json(nullptr)},
            {"domain_exhausted", info.domainExhausted},
            {"distinct_measured_count", info.distinctMeasuredCandidateCount},
            {"version", 1u},
            {"catalog", m_space.catalog()},
            {"global_candidates", m_space.globalCandidates()},
            {"cursor", m_space.cursor()},
            {"revision", m_space.revision()},
            {"state", static_cast<unsigned>(m_space.state())},
            {"random_state", m_space.randomState()}};
  }
  void updateCandidateSpaceCache(nlohmann::json &cache) const {
    if (!m_space.usesCandidateCatalog())
      return;
    auto &saved = cache["candidate_space"];
    if (!saved.is_object() || !saved.contains("catalog") ||
        saved["catalog"].size() != m_space.size() ||
        saved.value("state", std::numeric_limits<unsigned>::max()) !=
            static_cast<unsigned>(m_space.state())) {
      saved = serializedCandidateSpace();
    } else
      saved["distinct_measured_count"] =
          m_space.info().distinctMeasuredCandidateCount;
  }

  auto restoreCandidateSpace(nlohmann::json const &cache) -> bool {
    if (!m_space.usesCandidateCatalog())
      return !cache.contains("candidate_space");
    try {
      auto const &saved = cache.at("candidate_space");
      if (saved.at("version").get<unsigned>() != 1u)
        return false;
      auto rows =
          saved.at("catalog").template get<std::vector<CandidateIndices>>();
      if (rows.size() < m_space.size())
        return false;
      if (!std::equal(m_space.catalog().begin(), m_space.catalog().end(),
                      rows.begin()))
        return false;
      for (auto const &row : rows)
        if (!candidateAccepted(row) ||
            (m_launchValidator && !m_launchValidator(*this, row)))
          return false;
      auto const state = saved.at("state").get<unsigned>();
      if (state > static_cast<unsigned>(SpaceState::stalled))
        return false;
      auto restoredSpace = m_space;
      restoredSpace.restore(
          std::move(rows), saved.at("cursor").get<std::size_t>(),
          saved.at("revision").get<std::size_t>(),
          static_cast<SpaceState>(state),
          saved.at("global_candidates").get<std::vector<bool>>());
      restoredSpace.restoreRandomState(
          saved.at("random_state").get<std::string>());
      m_space = std::move(restoredSpace);
      resizeCandidateStorage();
      return true;
    } catch (std::exception const &) {
      return false;
    }
  }
#endif

  [[nodiscard]] auto loadHistory() -> bool {
#if ALPAKA_TUNE_HAS_JSON
    auto staged = m_history->stagedCache(m_fingerprint);
    auto store = nlohmann::json{};
    auto const *cachePointer = staged.get();
    auto stagedCache = cachePointer != nullptr;
    if (cachePointer == nullptr) {
      if (!m_history->file || !m_history->readFile)
        return false;
      std::lock_guard lock{m_history->mutex};
      auto const &path = *m_history->file;
      if (!std::filesystem::exists(path))
        return false;
      std::ifstream input{path};
      if (!(input >> store) || !store.contains("contexts") ||
          !store["contexts"].is_object())
        throw std::runtime_error{"The alpakaTune compact history is invalid."};
      if (store.value("schema_version", 0) != detail::historySchemaVersion)
        throw std::runtime_error{
            "The alpakaTune compact history uses an incompatible schema."};
      if (!store["contexts"].contains(m_fingerprint))
        return false;
      cachePointer = &store["contexts"][m_fingerprint];
    }
    auto const &cache = *cachePointer;
    if (!restoreCandidateSpace(cache))
      return false;
    auto restored = std::vector<std::tuple<std::size_t, double, std::size_t>>{};
    try {
      auto const &metric = cache.at("metric");
      if (metric.at("kind").get<std::string>() !=
              tuningMetricKindName(metricKind()) ||
          metric.at("name").get<std::string>() != m_metricName)
        return false;
      auto const &records =
          stagedCache ? cache.at("records") : cache.at("configurations");
      if (stagedCache) {
        if (!records.is_object())
          return false;
        restored.reserve(records.size());
        for (auto const &[key, record] : records.items()) {
          static_cast<void>(key);
          auto const candidate =
              candidateForCompactConfiguration(record.at("configuration"));
          if (!candidate)
            return false;
          restored.emplace_back(
              *candidate, record.at("median_metric_value").get<double>(),
              record.at("measurement_count").get<std::size_t>());
        }
      } else {
        if (!records.is_array())
          return false;
        restored.reserve(records.size());
        for (auto const &record : records) {
          auto const candidate =
              candidateForCompactConfiguration(record.at("configuration"));
          if (!candidate)
            return false;
          restored.emplace_back(
              *candidate, record.at("median_metric_value").get<double>(),
              record.at("measurement_count").get<std::size_t>());
        }
      }
      if (auto const adapter = cache.find("adapter");
          adapter != cache.end() && adapter->is_object())
        m_loadedLearnedAdapterState = parsedLearnedAdapter(*adapter, true);
    } catch (std::exception const &) {
      return false;
    }
    auto restoredCount = std::size_t{};
    for (auto const &[candidate, median, count] : restored) {
      if (m_loadedFromHistoryCandidates.at(candidate))
        continue;
      m_histories.at(candidate).restoreSummary(median, count);
      m_loadedFromHistoryCandidates.at(candidate) = true;
      ++restoredCount;
    }
    if (restoredCount == 0u)
      return false;
    m_retiredConfigurationCount = restoredCount;
    m_scheduled.assign(m_candidateCount, false);
    m_scheduledCount = 0u;
    if (m_defaults.selection == SelectionPolicy::fixed) {
      for (std::size_t candidate = 0u; candidate < m_candidateCount;
           ++candidate) {
        if (!m_histories.at(candidate).empty()) {
          m_scheduled.at(candidate) = true;
          ++m_scheduledCount;
        }
      }
    }
    auto const best = currentBestCandidate();
    if (!best)
      return false;
    m_bestCandidate = *best;
    m_bestRetiredRuntime = m_histories.at(*best).statistics().estimate();
    return true;
#else
    return false;
#endif
  }

  [[nodiscard]] auto loadCompleteHistory() -> bool {
#if ALPAKA_TUNE_HAS_JSON
    auto staged = m_completeHistory->stagedCache(completeHistorySchemaVersion,
                                                 m_fingerprint);
    auto store = nlohmann::json{};
    auto const *cachePointer = staged.get();
    if (cachePointer == nullptr) {
      if (!m_completeHistory->file || !m_completeHistory->readFile)
        return false;
      std::lock_guard lock{m_completeHistory->mutex};
      auto const &path = *m_completeHistory->file;
      if (!std::filesystem::exists(path))
        return false;
      std::ifstream input{path};
      if (!(input >> store) || !store.contains("contexts") ||
          !store["contexts"].is_object())
        throw std::runtime_error{"The alpakaTune complete history is invalid."};
      if (store.value("schema_version", 0) != completeHistorySchemaVersion)
        throw std::runtime_error{
            "The alpakaTune complete history uses an incompatible "
            "schema; fresh histories are required."};
      if (!store["contexts"].contains(m_fingerprint))
        return false;
      cachePointer = &store["contexts"][m_fingerprint];
    }
    auto const &cache = *cachePointer;
    if (!restoreCandidateSpace(cache))
      return false;
    if (cache.value("fingerprint", "") != m_fingerprint)
      return false;
    if (cache.value("candidate_count", std::size_t{}) != m_candidateCount)
      return false;
    auto const completionReason = completionReasonFromName(
        cache.value("completion_reason", std::string{"none"}));
    auto const storedExecutionCount =
        cache.value("execution_count", std::size_t{});
    auto measuredCount = static_cast<std::size_t>(
        std::count_if(m_histories.begin(), m_histories.end(),
                      [](auto const &history) { return !history.empty(); }));
    try {
      auto const &metric = cache.at("metadata").at("metric");
      if (metric.at("kind").get<std::string>() !=
              tuningMetricKindName(metricKind()) ||
          metric.at("name").get<std::string>() != m_metricName)
        return false;
      auto const &storedHistories = cache.at("candidate_samples");
      if (!storedHistories.is_array() ||
          storedHistories.size() != m_candidateCount)
        return false;
      auto const rejected =
          cache.at("rejected_candidates").get<std::vector<bool>>();
      auto const userInvalidated =
          cache.at("user_invalidated_candidates").get<std::vector<bool>>();
      auto const missingMetric =
          cache.at("missing_metric_candidates").get<std::vector<bool>>();
      if (rejected.size() != m_candidateCount ||
          userInvalidated.size() != m_candidateCount ||
          missingMetric.size() != m_candidateCount)
        return false;
      for (std::size_t candidate = 0u; candidate < m_candidateCount;
           ++candidate) {
        auto const samples =
            storedHistories.at(candidate).get<std::vector<double>>();
        if (rejected.at(candidate)) {
          if (!samples.empty() || m_loadedFromHistoryCandidates.at(candidate))
            return false;
          continue;
        }
        if (samples.empty())
          continue;
        if (!m_loadedFromHistoryCandidates.at(candidate)) {
          m_histories.at(candidate).restoreCompleted(samples);
          ++measuredCount;
        }
      }
      if (measuredCount == 0u)
        return false;
      m_rejected = rejected;
      m_rejectedCount = static_cast<std::size_t>(
          std::count(m_rejected.begin(), m_rejected.end(), true));
      m_userInvalidated = userInvalidated;
      m_userInvalidatedCount = static_cast<std::size_t>(
          std::count(m_userInvalidated.begin(), m_userInvalidated.end(), true));
      m_missingMetric = missingMetric;
      m_missingMetricCount = static_cast<std::size_t>(
          std::count(m_missingMetric.begin(), m_missingMetric.end(), true));
      for (std::size_t candidate = 0u; candidate < m_candidateCount;
           ++candidate) {
        if (!m_userInvalidated.at(candidate))
          continue;
        m_candidateValidity.at(candidate)->valid = false;
        m_candidateValidity.at(candidate)->invalidationConsumed = true;
      }
      m_scheduled.assign(m_candidateCount, false);
      m_scheduledCount = 0u;
      if (m_defaults.selection == SelectionPolicy::fixed) {
        for (std::size_t candidate = 0u; candidate < m_candidateCount;
             ++candidate) {
          if (m_rejected.at(candidate) || m_userInvalidated.at(candidate) ||
              m_missingMetric.at(candidate) ||
              m_histories.at(candidate).empty())
            continue;
          m_scheduled.at(candidate) = true;
          ++m_scheduledCount;
        }
      }

      m_retiredConfigurationCount =
          std::max(measuredCount,
                   cache.value("retired_configuration_count", measuredCount));
      if (auto const admission = cache.find("admission");
          admission != cache.end() && admission->is_object()) {
        m_unseenAcceptedCount = admission->value("unseen_accepted", 0u);
        m_revisitAcceptedCount = admission->value("revisit_accepted", 0u);
        m_activeDuplicateAcceptedCount =
            admission->value("active_duplicate_accepted", 0u);
        m_restrictionRejectedCount =
            admission->value("restriction_rejected", 0u);
        m_userInvalidatedRejectedCount =
            admission->value("user_invalidated_rejected", 0u);
        m_missingMetricRejectedCount =
            admission->value("missing_metric_rejected", 0u);
        m_revisitRejectedCount = admission->value("revisit_rejected", 0u);
        m_scoreRejectedCount = admission->value("score_rejected", 0u);
        m_strategyRetryLimitReachedCount =
            admission->value("strategy_retry_limit_reached", 0u);
        m_adaptiveRetryFallbackCount =
            admission->value("adaptive_retry_fallback", 0u);
      }
      m_bestImprovements.clear();
    } catch (std::exception const &) {
      return false;
    }
    auto const best = currentBestCandidate();
    if (!best)
      return false;
    m_bestCandidate = *best;
    m_bestRetiredRuntime = m_histories.at(*best).statistics().estimate();
    m_executionCount = storedExecutionCount;
    m_startedAtUnixSeconds = cache.value("started_at_unix_seconds", 0.0);
    loadCompleteLearnedAdapter(cache);
    if (m_defaults.exploration == ExplorationPolicy::offline) {
      m_terminal = true;
      m_completionReason = TunerCompletionReason::offlineReplay;
    } else if (m_defaults.selection == SelectionPolicy::fixed &&
               completionReason != TunerCompletionReason::none) {
      m_terminal = true;
      m_completionReason = completionReason;
      m_executionBudgetReached =
          completionReason == TunerCompletionReason::maximumExecutions;
    } else {
      m_terminal = false;
      m_completionReason = TunerCompletionReason::none;
    }
    return true;
#else
    return false;
#endif
  }

  void clearExplorationQueue() {
    m_queue.reset();
    m_scheduled.assign(m_candidateCount, false);
    m_scheduledCount = 0u;
    m_refinementPending = false;
  }

  void finishTuning() {
    auto best = std::optional<std::size_t>{};
    auto bestSeconds = std::numeric_limits<double>::infinity();
    for (std::size_t candidate = 0u; candidate < m_candidateCount;
         ++candidate) {
      if (m_rejected.at(candidate) || !candidateUsable(candidate))
        continue;
      auto const statistics = m_histories.at(candidate).statistics();
      if (statistics.sampleCount == 0u)
        throw std::logic_error{
            "A candidate completed without a measured launch."};
      auto const seconds = statistics.estimate();
      if (seconds < bestSeconds) {
        bestSeconds = seconds;
        best = candidate;
      }
    }
    m_bestCandidate = best.value_or(std::numeric_limits<std::size_t>::max());
    m_explorationComplete = true;
    m_terminal = m_defaults.selection == SelectionPolicy::fixed || !best;
    clearExplorationQueue();
    m_completionReason = best ? spaceCompletionReason()
                              : TunerCompletionReason::noValidConfiguration;
    writeCache();
  }

  void finishTuningAtBudget(TunerCompletionReason reason) {
    if (reason != TunerCompletionReason::maximumExecutions &&
        reason != TunerCompletionReason::maximumRetiredConfigurations &&
        reason != TunerCompletionReason::insufficientExpectedBenefit &&
        reason != TunerCompletionReason::tuningOverheadBudget)
      throw std::logic_error{"Invalid completion-limit reason."};
    auto const best = currentBestCandidate();
    if (!best)
      throw std::logic_error{"The tuning completion limit was reached before "
                             "any candidate was measured."};
    m_bestCandidate = *best;
    m_explorationComplete = true;
    m_terminal = m_defaults.selection == SelectionPolicy::fixed;
    m_completionReason = reason;
    m_executionBudgetReached =
        reason == TunerCompletionReason::maximumExecutions;
    clearExplorationQueue();
    writeCache();
  }

  void finishTuningAtStrategyRetryLimit() {
    if (auto const best = currentBestCandidate()) {
      m_bestCandidate = *best;
      m_completionReason =
          TunerCompletionReason::maximumConsecutiveStrategyRetries;
    } else {
      m_bestCandidate = std::numeric_limits<std::size_t>::max();
      m_completionReason =
          m_userInvalidatedCount == 0u && m_missingMetricCount == 0u
              ? TunerCompletionReason::maximumConsecutiveStrategyRetries
              : TunerCompletionReason::noValidConfiguration;
    }
    m_explorationComplete = true;
    m_terminal = m_defaults.selection == SelectionPolicy::fixed ||
                 !currentBestCandidate();
    clearExplorationQueue();
    writeCache();
  }

  /** @brief Update the in-memory JSON snapshot after one measured launch.
   *
   * Sparse candidate arrays are updated in place. No filesystem operation is
   * performed; CompleteHistoryStore flushes the newest snapshot at shutdown.
   */
  void stageCache(std::size_t candidate, bool schedulingChanged) {
#if ALPAKA_TUNE_HAS_JSON
    auto learning = std::optional<nlohmann::json>{};
    if (m_defaults.strategy == StrategyKind::learnedHybrid &&
        (schedulingChanged || !m_stagedCompleteHistory || !m_stagedHistory))
      learning = serializedLearningStatus();
    if (!m_stagedCompleteHistory ||
        m_stagedCompleteHistory->value("candidate_count", std::size_t{}) !=
            m_candidateCount) {
      m_stagedCompleteHistory =
          std::make_shared<nlohmann::json>(serializedCache(learning));
    } else {
      auto &cache = *m_stagedCompleteHistory;
      cache["execution_count"] = m_executionCount;
      cache["adaptive_horizon_execution_count"] =
          m_adaptiveHorizonExecutionCount;
      cache["retired_configuration_count"] = m_retiredConfigurationCount;
      if (auto const best = currentBestCandidate())
        cache["best_candidate_index"] = *best;
      auto samples = nlohmann::json::array();
      for (auto const seconds : m_histories.at(candidate).samples())
        samples.push_back(seconds);
      cache["candidate_samples"][candidate] = std::move(samples);
      if (m_histories.at(candidate).empty())
        cache["candidate_estimates"][candidate] = nullptr;
      else {
        cache["candidate_estimates"][candidate] =
            m_histories.at(candidate).statistics().estimate();
        cache["candidate_configurations"][candidate] =
            serializedCandidateConfiguration(candidate);
      }
      if (schedulingChanged)
        cache["rejected_candidates"] = m_rejected;
      if (m_defaults.strategy == StrategyKind::learnedHybrid &&
          schedulingChanged && learning)
        cache["learning"] = *learning;
      cache["admission"] = serializedAdmissionStatus();
      while (cache["best_improvements"].size() < m_bestImprovements.size())
        cache["best_improvements"].push_back(serializedImprovement(
            m_bestImprovements.at(cache["best_improvements"].size())));
    }
    updateCandidateSpaceCache(*m_stagedCompleteHistory);
    m_completeHistory->stageCache(completeHistorySchemaVersion, m_fingerprint,
                                  m_stagedCompleteHistory);
    stageHistoryCache(candidate, schedulingChanged, learning);
#endif
  }

#if ALPAKA_TUNE_HAS_JSON
  [[nodiscard]] auto
  serializedImprovement(detail::BestImprovement const &improvement) const
      -> nlohmann::json {
    return {
        {"candidate_index", improvement.candidateIndex},
        {"execution_count", improvement.executionCount},
        {"retired_configuration_count", improvement.retiredConfigurationCount},
        {"metric_value", improvement.metricValue},
        {"elapsed_seconds", improvement.elapsedSeconds},
        {"timestamp_unix_seconds",
         m_startedAtUnixSeconds + improvement.elapsedSeconds}};
  }

  template <typename Entry>
  void appendSerializedCandidateEntry(
      nlohmann::json &configuration,
      std::array<std::size_t, dimensionCount> const &indices) const {
    static_cast<void>(withCandidateValue<Entry::name>(
        indices, [&configuration](auto const &value) {
          configuration[std::string{Entry::name.view()}] =
              detail::printable(value);
          return true;
        }));
  }

  [[nodiscard]] auto
  serializedCandidateConfiguration(std::size_t candidate) const
      -> nlohmann::json {
    auto configuration = nlohmann::json::object();
    auto const indices = indicesFor(candidate);
    std::apply(
        [this, &configuration, &indices](auto const &...entries) {
          (appendSerializedCandidateEntry<
               std::remove_cvref_t<decltype(entries)>>(configuration, indices),
           ...);
        },
        m_tunables.entries());
    return configuration;
  }

  template <typename Value>
  [[nodiscard]] static auto serializedCompactValue(Value const &value)
      -> nlohmann::json {
    using Type = std::remove_cvref_t<Value>;
    if constexpr (alpaka::isVector_v<Type>) {
      auto result = nlohmann::json::array();
      for (std::size_t dimension = 0u; dimension < Type::dim(); ++dimension)
        result.push_back(serializedCompactValue(value[dimension]));
      return result;
    } else if constexpr (std::is_enum_v<Type>) {
      return static_cast<std::underlying_type_t<Type>>(value);
    } else if constexpr (std::is_arithmetic_v<Type>) {
      return value;
    } else {
      return detail::printable(value);
    }
  }

  template <typename Entry>
  void appendSerializedCompactCandidateEntry(
      nlohmann::json &configuration,
      std::array<std::size_t, dimensionCount> const &indices) const {
    static_cast<void>(withCandidateValue<Entry::name>(
        indices, [&configuration](auto const &value) {
          configuration[std::string{Entry::name.view()}] =
              serializedCompactValue(value);
          return true;
        }));
  }

  [[nodiscard]] auto
  serializedCompactCandidateConfiguration(std::size_t candidate) const
      -> nlohmann::json {
    auto configuration = nlohmann::json::object();
    auto const indices = indicesFor(candidate);
    std::apply(
        [this, &configuration, &indices](auto const &...entries) {
          (appendSerializedCompactCandidateEntry<
               std::remove_cvref_t<decltype(entries)>>(configuration, indices),
           ...);
        },
        m_tunables.entries());
    return configuration;
  }

  template <typename Entry>
  [[nodiscard]] auto
  resolveCompactEntry(nlohmann::json const &stored,
                      std::array<std::size_t, dimensionCount> &indices) const
      -> bool {
    using Values = typename Entry::values_type;
    constexpr auto offset = Traits::template dimensionOffset<Entry::name>;
    constexpr auto dimensions = detail::candidateDimensionCount<Values>;
    if constexpr (dimensions == 1u &&
                  requires { std::declval<Values>().minimum(); }) {
      auto const &entry =
          std::get<Traits::template index<Entry::name>>(m_tunables.entries());
      auto const index = entry.values.indexOf(
          stored.template get<typename Values::value_type>());
      if (!index)
        return false;
      indices[offset] = *index;
      return true;
    } else if constexpr (dimensions == 1u) {
      for (std::size_t index = 0u; index < m_dimensionSizes.at(offset);
           ++index) {
        indices[offset] = index;
        auto matches = false;
        static_cast<void>(withCandidateValue<Entry::name>(
            indices, [&stored, &matches](auto const &value) {
              matches = serializedCompactValue(value) == stored;
              return true;
            }));
        if (matches)
          return true;
      }
      return false;
    } else {
      if (!stored.is_array() || stored.size() != dimensions)
        return false;
      for (std::size_t dimension = 0u; dimension < dimensions; ++dimension) {
        auto matched = false;
        for (std::size_t index = 0u;
             index < m_dimensionSizes.at(offset + dimension); ++index) {
          indices[offset + dimension] = index;
          auto componentMatches = false;
          static_cast<void>(withCandidateValue<Entry::name>(
              indices,
              [&stored, &componentMatches, dimension](auto const &value) {
                componentMatches = serializedCompactValue(value[dimension]) ==
                                   stored.at(dimension);
                return true;
              }));
          if (componentMatches) {
            matched = true;
            break;
          }
        }
        if (!matched)
          return false;
      }
      return true;
    }
  }

  template <typename Entry>
  [[nodiscard]] auto resolveCompactConfigurationEntry(
      nlohmann::json const &configuration,
      std::array<std::size_t, dimensionCount> &indices) const -> bool {
    auto const found = configuration.find(std::string{Entry::name.view()});
    return found != configuration.end() &&
           resolveCompactEntry<Entry>(*found, indices);
  }

  [[nodiscard]] auto
  candidateForCompactConfiguration(nlohmann::json const &configuration) const
      -> std::optional<std::size_t> {
    if (!configuration.is_object() ||
        configuration.size() != std::tuple_size_v<Entries>)
      return std::nullopt;
    auto indices = std::array<std::size_t, dimensionCount>{};
    auto valid = true;
    std::apply(
        [this, &configuration, &indices, &valid](auto const &...entries) {
          ((valid = resolveCompactConfigurationEntry<
                        std::remove_cvref_t<decltype(entries)>>(configuration,
                                                                indices) &&
                    valid),
           ...);
        },
        m_tunables.entries());
    if (!valid || !candidateAccepted(indices))
      return std::nullopt;
    auto const candidate = m_space.find(indices);
    if (!candidate ||
        serializedCompactCandidateConfiguration(*candidate) != configuration)
      return std::nullopt;
    return candidate;
  }

  [[nodiscard]] auto serializedLearnedModelContext() const -> nlohmann::json {
    auto const descriptor = learnedModelContext();
    auto result = nlohmann::json{
        {"feature_schema_version", descriptor.schemaVersion},
        {"workload_id", modelWorkloadId()},
        {"device_class",
         descriptor.deviceClass == LearnedDeviceClass::gpu ? "gpu" : "cpu"},
        {"context_features", nlohmann::json::object()},
        {"dimensions", nlohmann::json::array()},
    };
    for (auto const &feature : descriptor.contextFeatures)
      result["context_features"][feature.name] = feature.value;
    for (auto const &dimension : descriptor.dimensions) {
      auto const kind = [&]() -> std::string_view {
        switch (dimension.kind) {
        case LearnedDimensionKind::runtime:
          return "runtime";
        case LearnedDimensionKind::compileTime:
          return "compile_time";
        case LearnedDimensionKind::launch:
          return "launch";
        case LearnedDimensionKind::categorical:
          return "categorical";
        }
        return "categorical";
      }();
      result["dimensions"].push_back(
          {{"name", dimension.name},
           {"kind", kind},
           {"cardinality", dimension.cardinality},
           {"component_index", dimension.componentIndex},
           {"vector_arity", dimension.vectorArity},
           {"concrete_values", dimension.concreteValues}});
    }
    return result;
  }

  [[nodiscard]] auto parsedLearnedAdapter(nlohmann::json const &serialized,
                                          bool compact) const
      -> std::optional<LearnedResidualAdapterState> {
    if (m_defaults.strategy != StrategyKind::learnedHybrid)
      return std::nullopt;
    try {
      auto const &learning = compact ? serialized : serialized.at("learning");
      if (learning.value("model_digest", std::string{}) != m_learnedModelDigest)
        return std::nullopt;
      auto const &adapter =
          compact ? learning : learning.at("residual_adapter");
      if (adapter.value("state_version", 0) != 1)
        return std::nullopt;
      auto state = LearnedResidualAdapterState{
          .coefficients = adapter.at("coefficients").get<std::vector<double>>(),
          .observationsSinceUpdate =
              adapter.value("observations_since_update", std::size_t{}),
          .updateCount = adapter.value("update_count", std::size_t{})};
      for (auto const &observation : adapter.at("observations"))
        state.observations.push_back(
            {.rawIndex =
                 observation.at("raw_candidate_index").get<std::size_t>(),
             .features = observation.at("features").get<std::vector<float>>(),
             .residual = observation.at("residual").get<double>()});
      return state;
    } catch (std::exception const &) {
      return std::nullopt;
    }
  }

  void loadCompleteLearnedAdapter(nlohmann::json const &cache) {
    auto state = parsedLearnedAdapter(cache, false);
    if (!state)
      return;
    if (m_loadedLearnedAdapterState)
      m_loadedCompleteLearnedAdapterState = std::move(state);
    else
      m_loadedLearnedAdapterState = std::move(state);
  }

  [[nodiscard]] auto serializedLearningStatus() const -> nlohmann::json {
    auto result = nlohmann::json{
        {"requested_strategy", std::string{strategyName(m_defaults.strategy)}},
        {"model_file", m_defaults.learnedModelFile.string()},
        {"model_digest", m_learnedModelDigest},
        {"fallback_strategy",
         std::string{strategyName(m_defaults.learnedFallback)}}};
    auto const *learned =
        dynamic_cast<LearnedHybridStrategy const *>(m_strategy.get());
    auto const *stored = m_loadedLearnedAdapterState
                             ? &*m_loadedLearnedAdapterState
                         : m_loadedCompleteLearnedAdapterState
                             ? &*m_loadedCompleteLearnedAdapterState
                             : nullptr;
    if (learned == nullptr && stored == nullptr)
      return result;
    if (learned != nullptr) {
      result["status"] = learnedHybridStatusName(learned->status());
      result["artifact_load_status"] =
          learnedModelLoadStatusName(learned->artifactLoadStatus());
      result["status_message"] = learned->statusMessage();
      result["initialization_seconds"] = learned->initializationSeconds();
      result["cached_candidate_count"] = learned->cachedCandidateCount();
      result["candidate_pool_capacity"] = learned->candidatePoolCapacity();
      result["candidate_batch_size"] = learned->candidateBatchSize();
      result["peak_cached_candidate_count"] =
          learned->peakCachedCandidateCount();
      result["scored_candidate_count"] = learned->scoredCandidateCount();
      result["pool_refill_count"] = learned->poolRefillCount();
      result["candidate_stream_exhausted"] =
          learned->candidateStreamExhausted();
      result["incorporated_observation_count"] =
          learned->incorporatedObservationCount();
      result["adapter_update_count"] = learned->adapterUpdateCount();
    } else {
      result["status"] = "known_reuse";
      result["adapter_update_count"] = stored->updateCount;
    }
    auto const state =
        learned != nullptr ? learned->residualAdapterState() : *stored;
    auto adapter = nlohmann::json{
        {"state_version", 1},
        {"coefficients", state.coefficients},
        {"observations_since_update", state.observationsSinceUpdate},
        {"update_count", state.updateCount},
        {"observations", nlohmann::json::array()}};
    for (auto const &observation : state.observations)
      adapter["observations"].push_back(
          {{"raw_candidate_index", observation.rawIndex},
           {"features", observation.features},
           {"residual", observation.residual}});
    result["residual_adapter"] = std::move(adapter);
    return result;
  }

  [[nodiscard]] auto
  serializedCompactAdapter(nlohmann::json const &learning) const
      -> std::optional<nlohmann::json> {
    if (m_defaults.strategy != StrategyKind::learnedHybrid)
      return std::nullopt;
    auto const status = learning.value("status", std::string{});
    if ((status != "active" && status != "known_reuse") ||
        !learning.contains("residual_adapter"))
      return std::nullopt;
    auto adapter = learning.at("residual_adapter");
    adapter["model_digest"] = learning.at("model_digest");
    return adapter;
  }

  [[nodiscard]] auto serializedAdmissionStatus() const -> nlohmann::json {
    return {{"unseen_accepted", m_unseenAcceptedCount},
            {"revisit_accepted", m_revisitAcceptedCount},
            {"active_duplicate_accepted", m_activeDuplicateAcceptedCount},
            {"restriction_rejected", m_restrictionRejectedCount},
            {"user_invalidated_rejected", m_userInvalidatedRejectedCount},
            {"missing_metric_rejected", m_missingMetricRejectedCount},
            {"revisit_rejected", m_revisitRejectedCount},
            {"score_rejected", m_scoreRejectedCount},
            {"strategy_retry_limit_reached", m_strategyRetryLimitReachedCount},
            {"adaptive_retry_fallback", m_adaptiveRetryFallbackCount},
            {"consecutive_strategy_retries", m_consecutiveStrategyRetries}};
  }

  [[nodiscard]] auto historySamplingSeed() const noexcept -> std::uint64_t {
    return detail::contextSeed(m_effectiveRandomSeed, m_fingerprint);
  }

  [[nodiscard]] auto
  serializedHistoryCache(std::optional<nlohmann::json> const &learning) const
      -> nlohmann::json {
    auto cache = nlohmann::json{
        {"records", nlohmann::json::object()},
        {"metric",
         {{"kind", tuningMetricKindName(metricKind())},
          {"name", m_metricName}}},
        {"sampling_seed", historySamplingSeed()},
        {"sample_count", m_defaults.history.sampleCount
                             ? nlohmann::json(*m_defaults.history.sampleCount)
                             : nlohmann::json(nullptr)}};
    for (std::size_t candidate = 0u; candidate < m_candidateCount;
         ++candidate) {
      if (!candidateUsable(candidate) || m_histories.at(candidate).empty())
        continue;
      auto const statistics = m_histories.at(candidate).statistics();
      cache["records"][std::to_string(candidate)] = {
          {"candidate_index", candidate},
          {"configuration", serializedCompactCandidateConfiguration(candidate)},
          {"median_metric_value", statistics.estimate()},
          {"measurement_count", statistics.sampleCount}};
    }
    if (m_space.usesCandidateCatalog())
      cache["candidate_space"] = serializedCandidateSpace();
    if (learning)
      if (auto adapter = serializedCompactAdapter(*learning))
        cache["adapter"] = std::move(*adapter);
    return cache;
  }

  void stageHistoryCache(std::size_t candidate, bool schedulingChanged,
                         std::optional<nlohmann::json> const &learning) {
    if (!m_stagedHistory)
      m_stagedHistory =
          std::make_shared<nlohmann::json>(serializedHistoryCache(learning));
    else {
      auto &cache = *m_stagedHistory;
      auto const statistics = m_histories.at(candidate).statistics();
      cache["records"][std::to_string(candidate)] = {
          {"candidate_index", candidate},
          {"configuration", serializedCompactCandidateConfiguration(candidate)},
          {"median_metric_value", statistics.estimate()},
          {"measurement_count", statistics.sampleCount}};
      if (schedulingChanged) {
        if (learning) {
          if (auto adapter = serializedCompactAdapter(*learning))
            cache["adapter"] = std::move(*adapter);
          else
            cache.erase("adapter");
        } else {
          cache.erase("adapter");
        }
      }
    }
    updateCandidateSpaceCache(*m_stagedHistory);
    m_history->stageCache(m_fingerprint, m_stagedHistory);
  }

  [[nodiscard]] auto
  serializedCache(std::optional<nlohmann::json> const &learning) const
      -> nlohmann::json {
    nlohmann::json cache;
    cache["fingerprint"] = m_fingerprint;
    cache["candidate_count"] = m_candidateCount;
    if (m_space.usesCandidateCatalog())
      cache["candidate_space"] = serializedCandidateSpace();
    if (auto const best = currentBestCandidate())
      cache["best_candidate_index"] = *best;
    else
      cache["best_candidate_index"] = nullptr;
    cache["execution_count"] = m_executionCount;
    cache["adaptive_horizon_execution_count"] = m_adaptiveHorizonExecutionCount;
    cache["retired_configuration_count"] = m_retiredConfigurationCount;
    cache["execution_budget_reached"] = m_executionBudgetReached;
    cache["completion_reason"] = completionReasonName(m_completionReason);
    cache["started_at_unix_seconds"] = m_startedAtUnixSeconds;
    if (m_terminal)
      cache["completed_at_unix_seconds"] =
          std::chrono::duration<double>{
              std::chrono::system_clock::now().time_since_epoch()}
              .count();
    cache["metadata"] = {
        {"identity_entries", m_identityEntries},
        {"kernel", m_kernelName},
        {"device", detail::identityName(m_device)},
        {"launch_specification", m_launchSpecification},
        {"metric",
         {{"kind", tuningMetricKindName(metricKind())},
          {"name", m_metricName}}},
        {"exploration",
         std::string{explorationPolicyName(m_defaults.exploration)}},
        {"selection", std::string{selectionPolicyName(m_defaults.selection)}},
        {"strategy", std::string{strategyName(m_defaults.strategy)}},
        {"model_context", serializedLearnedModelContext()}};
    cache["exploration_complete"] = m_explorationComplete;
    cache["policy"] = {
        {"adaptive_probe_interval", m_defaults.adaptiveProbeInterval},
        {"history_window_size", m_defaults.historyWindowSize},
        {"maximum_consecutive_strategy_retries",
         m_defaults.maximumConsecutiveStrategyRetries}};
    if (m_space.usesCandidateCatalog())
      cache["policy"]["space"] = {
          {"initial_candidates", m_defaults.space.initialCandidates},
          {"refinement_batch_size", m_defaults.space.refinementBatchSize},
          {"refinement_interval", m_defaults.space.refinementInterval},
          {"maximum_candidates", m_defaults.space.maximumCandidates},
          {"maximum_generation_attempts",
           m_defaults.space.maximumGenerationAttempts},
          {"plateau_patience",
           m_defaults.space.plateauPatience
               ? nlohmann::json(*m_defaults.space.plateauPatience)
               : nlohmann::json(nullptr)},
          {"minimum_relative_improvement",
           m_defaults.space.minimumRelativeImprovement}};
    cache["policy"]["queue"] = nullptr;
    if (m_defaults.queue) {
      cache["policy"]["queue"] = {
          {"disable", m_defaults.queue->disable},
          {"warmup_runs", m_defaults.queue->warmupRuns},
          {"noise_cancellation_window",
           m_defaults.queue->noiseCancellationWindow},
          {"max_consecutive_runs", m_defaults.queue->maxConsecutiveRuns}};
    }
    if (m_defaults.selection == SelectionPolicy::adaptive) {
      cache["policy"]["horizon"] = m_defaults.horizon
                                       ? nlohmann::json{*m_defaults.horizon}
                                       : nlohmann::json{nullptr};
      cache["policy"]["revisit_admission_steepness"] =
          m_defaults.revisitAdmissionSteepness;
      cache["policy"]["score_temperature_start"] =
          m_defaults.scoreTemperatureStart;
      cache["policy"]["score_temperature_end"] = m_defaults.scoreTemperatureEnd;
      cache["policy"]["horizon_offset_with_active_history"] =
          m_defaults.horizonOffsetWithActiveHistory;
    }
    cache["admission"] = serializedAdmissionStatus();
    if (learning)
      cache["learning"] = *learning;
    if (m_defaults.exploration == ExplorationPolicy::online) {
      if (m_defaults.maximumExecutions)
        cache["limits"]["maximum_executions"] = *m_defaults.maximumExecutions;
      else
        cache["limits"]["maximum_executions"] = nullptr;
      if (m_defaults.maximumRetiredConfigurations)
        cache["limits"]["maximum_retired_configurations"] =
            *m_defaults.maximumRetiredConfigurations;
      else
        cache["limits"]["maximum_retired_configurations"] = nullptr;
    }
    cache["best_improvements"] = nlohmann::json::array();
    for (auto const &improvement : m_bestImprovements)
      cache["best_improvements"].push_back(serializedImprovement(improvement));
    cache["rejected_candidates"] = m_rejected;
    cache["user_invalidated_candidates"] = m_userInvalidated;
    cache["missing_metric_candidates"] = m_missingMetric;
    cache["candidate_samples"] = nlohmann::json::array();
    cache["candidate_estimates"] = nlohmann::json::array();
    cache["candidate_configurations"] = nlohmann::json::array();
    for (std::size_t candidate = 0u; candidate < m_candidateCount;
         ++candidate) {
      auto samples = nlohmann::json::array();
      for (auto const seconds : m_histories.at(candidate).samples())
        samples.push_back(seconds);
      cache["candidate_samples"].push_back(std::move(samples));
      if (m_histories.at(candidate).empty())
        cache["candidate_estimates"].push_back(nullptr);
      else
        cache["candidate_estimates"].push_back(
            m_histories.at(candidate).statistics().estimate());
      if (m_histories.at(candidate).empty())
        cache["candidate_configurations"].push_back(nullptr);
      else
        cache["candidate_configurations"].push_back(
            serializedCandidateConfiguration(candidate));
    }
    return cache;
  }
#endif

  /** @brief Stage a complete snapshot without writing the persistence file. */
  void writeCache() {
#if ALPAKA_TUNE_HAS_JSON
    auto learning = std::optional<nlohmann::json>{};
    if (m_defaults.strategy == StrategyKind::learnedHybrid)
      learning = serializedLearningStatus();
    if (!m_stagedCompleteHistory)
      m_stagedCompleteHistory = std::make_shared<nlohmann::json>();
    *m_stagedCompleteHistory = serializedCache(learning);
    m_completeHistory->stageCache(completeHistorySchemaVersion, m_fingerprint,
                                  m_stagedCompleteHistory);
    if (!m_stagedHistory)
      m_stagedHistory = std::make_shared<nlohmann::json>();
    *m_stagedHistory = serializedHistoryCache(learning);
    updateCandidateSpaceCache(*m_stagedHistory);
    m_history->stageCache(m_fingerprint, m_stagedHistory);
#endif
  }

  std::optional<detail::BenefitBudget> m_budget;
  std::optional<std::size_t> m_budgetIncumbent;
  std::optional<std::size_t> m_budgetDirectCandidate;
  std::vector<bool> m_budgetInferior;
  std::vector<std::size_t> m_budgetVisitSamples;
  std::vector<std::size_t> m_budgetWarmups;
  std::vector<detail::RuntimeStatistics> m_budgetStatistics;
  std::optional<TunerCompletionReason> m_budgetStoppingReason;
  std::uint64_t m_budgetNextCheck{};
  std::size_t m_budgetBootstrapSamples{};
  bool m_budgetColdStart{};
  bool m_budgetEpochChanged{};
  double m_budgetLastDriftValue{};
  std::vector<bool> m_budgetEpochFresh;
  std::size_t m_budgetDriftSamples{};
  detail::RuntimeStatistics m_budgetCheckReference;
  LaunchPurpose m_budgetPurpose{LaunchPurpose::production};
  double m_budgetPendingHostCost{};
  double m_budgetPendingIncumbent{};
  void const *m_budgetContextTag{};
  std::function<void(void const *)> m_budgetContextValidator;
  std::size_t m_budgetCachedDispatch{};
  std::size_t m_budgetDispatchCandidate{
      std::numeric_limits<std::size_t>::max()};
  std::shared_ptr<std::uint64_t> m_validityGeneration{
      std::make_shared<std::uint64_t>(0u)};
  std::uint64_t m_consumedValidityGeneration{};
  TunerConfig m_defaults;
  std::shared_ptr<detail::HistoryStore> m_history;
  std::shared_ptr<detail::CompleteHistoryStore> m_completeHistory;
  TunablesType m_tunables;
  Device m_device;
  detail::MeasurementTimerStorage<Device, !usesCustomMetric> m_measurementTimer;
  std::vector<std::string> m_identityEntries;
  std::string m_metricName;
  [[no_unique_address]] MetricPolicy m_metricPolicy;
  std::vector<std::size_t> m_dimensionSizes;
  detail::CandidateSpace<dimensionCount> m_space;
  std::vector<std::function<bool(
      Tuner const &, std::array<std::size_t, dimensionCount> &, bool)>>
      m_domainProjectors;
  bool m_refinementPending{};
  std::function<bool(Tuner const &, CandidateIndices const &)>
      m_launchValidator;
  std::size_t m_candidateCount{1u};
  std::uint64_t m_baseRandomSeed{};
  std::uint64_t m_effectiveRandomSeed{};
  std::mt19937_64 m_random;
  std::uniform_real_distribution<double> m_admissionDistribution{0.0, 1.0};
  std::unique_ptr<ParameterStrategy> m_strategy;
  std::optional<LearnedResidualAdapterState> m_loadedLearnedAdapterState;
  std::optional<LearnedResidualAdapterState>
      m_loadedCompleteLearnedAdapterState;
  std::unique_ptr<detail::CandidateQueue> m_queue;
  std::vector<bool> m_scheduled;
  std::vector<bool> m_rejected;
  std::vector<bool> m_userInvalidated;
  std::vector<bool> m_missingMetric;
  std::vector<std::shared_ptr<detail::ConfigurationValidityState>>
      m_candidateValidity;
  std::vector<bool> m_loadedFromHistoryCandidates;
  std::size_t m_scheduledCount{};
  std::size_t m_rejectedCount{};
  std::size_t m_userInvalidatedCount{};
  std::size_t m_missingMetricCount{};
  std::size_t m_executionCount{};
  /** Executions in this adaptive process run; intentionally not restored. */
  std::size_t m_adaptiveHorizonExecutionCount{};
  std::size_t m_retiredConfigurationCount{};
  std::size_t m_unseenAcceptedCount{};
  std::size_t m_revisitAcceptedCount{};
  std::size_t m_activeDuplicateAcceptedCount{};
  std::size_t m_restrictionRejectedCount{};
  std::size_t m_userInvalidatedRejectedCount{};
  std::size_t m_missingMetricRejectedCount{};
  std::size_t m_revisitRejectedCount{};
  std::size_t m_scoreRejectedCount{};
  std::size_t m_consecutiveStrategyRetries{};
  std::size_t m_strategyRetryLimitReachedCount{};
  std::size_t m_adaptiveRetryFallbackCount{};
  double m_bestRetiredRuntime{std::numeric_limits<double>::infinity()};
  std::vector<detail::BestImprovement> m_bestImprovements;
  std::chrono::steady_clock::time_point m_tuningStarted{};
  double m_startedAtUnixSeconds{};
  std::vector<detail::RuntimeHistory> m_histories;
  std::vector<ExecutedConfiguration> m_executionHistory;
  std::optional<PendingMetric> m_pendingMetric;
  std::vector<CompileVariantFunctor> m_compileVariants;
  std::vector<std::function<bool(Tuner const &, void const *, void const *,
                                 CandidateIndices const &)>>
      m_compileValidators;
  std::unordered_map<std::string, std::size_t> m_compileVariantIndices;
  std::string m_compileVariantSignature;
  std::string m_fingerprint;
  std::string m_learnedModelDigest{"unavailable"};
  std::string m_kernelName;
  std::string m_launchSpecification;
  std::size_t m_bestCandidate{std::numeric_limits<std::size_t>::max()};
  double m_recommendationSecondsSinceLastLaunch{};
  TunerCompletionReason m_completionReason{TunerCompletionReason::none};
  /** Terminal replay state, intentionally independent of adaptive horizon. */
  bool m_explorationComplete{};
  std::size_t m_reuseLaunchCount{};
  std::size_t m_nextProbeCandidate{};
  bool m_terminal{};
  bool m_schedulingInitialised{};
  bool m_loadedFromCache{};
  bool m_executionBudgetReached{};
  std::optional<InstrumentationOverheadWarning>
      m_instrumentationOverheadWarning;
#if ALPAKA_TUNE_HAS_JSON
  std::shared_ptr<nlohmann::json> m_stagedHistory;
  std::shared_ptr<nlohmann::json> m_stagedCompleteHistory;
#endif
};

/** @brief Construct a device-bound tuner from an explicit policy.
 *
 * @param config Tuner policy, validated before construction.
 * @param tunables Runtime, compile-time, and launch-shape candidate bundle.
 * @param device Physical device to which subsequent queues must belong.
 * @param identityEntries Additional stable workload identity components and,
 * optionally, one customMetric() compile-time policy token.
 *
 * Identity-entry order does not affect the persistent fingerprint.
 */
template <typename TunablesType, typename Device, typename... IdentityEntries>
  requires detail::isTunableBundle<TunablesType>
[[nodiscard]] auto makeTuner(TunerConfig config, TunablesType tunables,
                             Device device,
                             IdentityEntries const &...identityEntries) {
  static_assert(detail::customMetricTokenCount<IdentityEntries...> <= 1u,
                "makeTuner accepts at most one customMetric() token.");
  using MetricPolicy = detail::SelectedMetricPolicy<IdentityEntries...>;
  auto metricPolicy = [&] {
    if constexpr (std::same_as<MetricPolicy, metric::Timing>)
      return metric::Timing{};
    else
      return std::get<MetricPolicy const &>(std::tie(identityEntries...));
  }();
  config.validate();
  auto names = std::vector<std::string>{};
  names.reserve(sizeof...(IdentityEntries));
  auto metricName = std::string{"runtime_seconds"};
  auto consumeEntry = [&](auto const &entry) {
    using Entry = std::remove_cvref_t<decltype(entry)>;
    if constexpr (detail::isCustomMetricToken<Entry>)
      metricName = entry.name;
    else
      names.push_back(detail::typeName<Entry>() + "=" +
                      detail::identityName(entry));
  };
  (consumeEntry(identityEntries), ...);
  std::sort(names.begin(), names.end());
  auto history = detail::historyStore(config.history.file, config.history.read,
                                      config.history.write);
  auto completeHistory = detail::completeHistoryStore(
      config.completeHistory.file, config.completeHistory.read,
      config.completeHistory.write);
  return Tuner<TunablesType, Device, MetricPolicy>{
      std::move(config),     std::move(history),     std::move(completeHistory),
      std::move(tunables),   std::move(device),      std::move(names),
      std::move(metricName), std::move(metricPolicy)};
}

/** @brief Construct a tuner using the process default YAML configuration. */
template <typename TunablesType, typename Device, typename... IdentityEntries>
  requires detail::isTunableBundle<TunablesType>
[[nodiscard]] auto makeTuner(TunablesType tunables, Device device,
                             IdentityEntries const &...identityEntries) {
  return makeTuner(tunerConfig(), std::move(tunables), std::move(device),
                   identityEntries...);
}

} // namespace alpakaTune
