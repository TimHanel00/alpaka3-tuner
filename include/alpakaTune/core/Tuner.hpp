// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "alpakaTune/adjust/DerivedThreadSpec.hpp"
#include "alpakaTune/core/Persistence.hpp"
#include "alpakaTune/core/TunerConfig.hpp"
#include "alpakaTune/core/TunerInfo.hpp"
#include "alpakaTune/core/peripherals/CandidateQueue.hpp"
#include "alpakaTune/model/LearnedModelContext.hpp"
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
  double runtimeSeconds{};
  double elapsedSeconds{};
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
[[nodiscard]] inline auto
adaptiveProgressWithActiveHistory(double progress, bool activeHistory,
                                  double offset) -> double {
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
template <typename TunablesType, typename Device> class Tuner {
public:
  using Traits = detail::TunablesTraits<TunablesType>;
  using Entries = typename Traits::entries_type;
  static constexpr std::size_t dimensionCount = Traits::dimensionCount;
  // The executor is part of the launch identity. Histories produced before
  // that identity was persisted must not be reused by a newer tuner.
  static constexpr int persistenceSchemaVersion = 11;
  static constexpr bool tunesFrameSpec =
      Traits::template has<detail::numFramesName> ||
      Traits::template has<detail::frameExtentName>;
  static constexpr bool tunesThreadSpec =
      Traits::template has<detail::numBlocksName> ||
      Traits::template has<detail::numThreadsName>;
  static_assert(!(tunesFrameSpec && tunesThreadSpec),
                "A tuner cannot combine FrameSpec and ThreadSpec parameters.");

  /** @brief Construct a tuner from a validated policy and identity components.
   *
   * Prefer makeTuner(); this constructor is public to preserve the aggregate
   * factory's concrete return type.
   */
  Tuner(TunerConfig defaults,
        std::shared_ptr<detail::PersistenceStore> persistence,
        TunablesType tunables, Device device,
        std::vector<std::string> identityEntries)
      : m_defaults(std::move(defaults)), m_persistence(std::move(persistence)),
        m_tunables(std::move(tunables)), m_device(std::move(device)),
        m_identityEntries(std::move(identityEntries)),
        m_random(m_defaults.randomSeed) {
    initialiseDimensions();
  }

  /** Whether tuning entered an actual terminal state.
   *
   * This remains false in online-adaptive mode. Use completed() only when the
   * application also wants the adaptive schedule-horizon diagnostic.
   */
  [[nodiscard]] auto isTuningComplete() const noexcept -> bool {
    return m_complete;
  }
  /** Whether the configured tuning-policy goal has been reached.
   *
   * In online-fixed and offline mode this is a terminal state. In
   * online-adaptive mode it only reports that the admission/cooling horizon
   * has been reached; enqueue() continues adapting normally afterward.
   */
  [[nodiscard]] auto completed() const noexcept -> bool {
    if (m_defaults.mode == TuningMode::onlineAdaptive)
      return adaptiveHorizonReached();
    return m_complete;
  }
  /** @brief Explain why an actual terminal state was entered.
   *
   * This is a post-completion diagnostic, not a configuration input or a
   * general status field.
   *
   * @throws std::logic_error if isTuningComplete() is false. Reaching the
   * adaptive schedule horizon does not create a terminal reason.
   */
  [[nodiscard]] auto completionReason() const -> TunerCompletionReason {
    if (!m_complete)
      throw std::logic_error{
          "The tuner has not entered a terminal completion state."};
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
  /** @brief Return the terminal winner's Cartesian index.
   * @throws std::logic_error before fixed/offline terminal completion.
   */
  [[nodiscard]] auto bestCandidateIndex() const -> std::size_t {
    if (!m_complete)
      throw std::logic_error{"The tuner has not selected a winner yet."};
    return m_bestCandidate;
  }
  /** @brief Return the terminal winner's normalized parameter vector. */
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
  /** @brief Cartesian index used by the most recent enqueue call. */
  [[nodiscard]] auto lastCandidateIndex() const noexcept -> std::size_t {
    return m_lastCandidate;
  }
  /** @brief Snapshot current coverage, policy, best, and admission counters. */
  [[nodiscard]] auto info() const -> TunerInfo {
    auto measured = std::size_t{};
    for (std::size_t candidate = 0u; candidate < m_histories.size();
         ++candidate)
      if (!m_rejected.empty() && !m_rejected.at(candidate) &&
          !m_histories.at(candidate).empty())
        ++measured;
    auto result = TunerInfo{
        .mode = m_defaults.mode,
        .candidateCount = m_candidateCount,
        .rejectedCandidateCount = m_rejectedCount,
        .scheduledCandidateCount = m_scheduledCount,
        .measuredCandidateCount = measured,
        .retiredConfigurationCount = m_retiredConfigurationCount,
        .executionCount = m_executionCount,
        .adaptiveHorizonExecutionCount = m_adaptiveHorizonExecutionCount,
        .adaptiveHorizonProgress =
            m_defaults.mode == TuningMode::onlineAdaptive ? adaptiveProgress()
                                                          : 0.0,
        .horizon = m_defaults.mode == TuningMode::onlineAdaptive
                       ? std::optional<std::size_t>{m_defaults.horizon}
                       : std::nullopt,
        .maximumExecutions = m_defaults.maximumExecutions,
        .tuningComplete = m_complete,
        .loadedFromCache = m_loadedFromCache,
        .executionBudgetReached = m_executionBudgetReached,
        .bestCandidateIndex = std::nullopt,
        .bestConfiguration = std::nullopt,
        .learnedStatus = std::nullopt,
        .learnedAdapterUpdateCount = 0u,
        .unseenAcceptedCount = m_unseenAcceptedCount,
        .revisitAcceptedCount = m_revisitAcceptedCount,
        .activeDuplicateRejectedCount = m_activeDuplicateRejectedCount,
        .restrictionRejectedCount = m_restrictionRejectedCount,
        .revisitRejectedCount = m_revisitRejectedCount,
        .scoreRejectedCount = m_scoreRejectedCount};
    if (auto const best = currentBestCandidate()) {
      result.bestCandidateIndex = *best;
      result.bestConfiguration = normalizedConfiguration(*best);
    } else if (m_bestCandidate != std::numeric_limits<std::size_t>::max()) {
      result.bestCandidateIndex = m_bestCandidate;
      result.bestConfiguration = normalizedConfiguration(m_bestCandidate);
    }
    if (auto const *learned =
            dynamic_cast<LearnedHybridStrategy const *>(m_strategy.get())) {
      result.learnedStatus = learnedHybridStatusName(learned->status());
      result.learnedAdapterUpdateCount = learned->adapterUpdateCount();
    }
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

  template <typename Queue, typename FrameSpec, typename Kernel,
            typename... Args>
  /** @brief Select and perform exactly one kernel launch.
   *
   * The application owns call count and lifetime. Depending on mode and queue
   * state, the launch may be a warm-up, measurement, adaptive revisit, or
   * terminal winner replay.
   */
  void enqueue(Queue const &queue, FrameSpec const &frameSpec,
               alpaka::KernelBundle<Kernel, Args...> const &prototype) {
    static_cast<void>(enqueueObserved(queue, frameSpec, prototype));
  }

  template <typename Queue, typename FrameSpec, typename Kernel,
            typename... Args>
  /** @brief Perform one launch and return its selection/timing diagnostics. */
  [[nodiscard]] auto
  enqueueObserved(Queue const &queue, FrameSpec const &frameSpec,
                  alpaka::KernelBundle<Kernel, Args...> const &prototype)
      -> LaunchObservation {
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
    initialiseScheduling();

    if (m_defaults.mode == TuningMode::onlineFixed && !m_complete &&
        executionBudgetReached())
      finishTuningAtBudget(TunerCompletionReason::maximumExecutions);
    if (m_defaults.mode == TuningMode::onlineFixed && !m_complete &&
        retiredConfigurationBudgetReached())
      finishTuningAtBudget(TunerCompletionReason::maximumRetiredConfigurations);

    using Bundle = std::remove_cvref_t<decltype(prototype)>;
    initialiseCompileVariants<Queue, FrameSpec, Bundle>();
    auto const selection =
        m_complete ? detail::CandidateQueue::Selection{m_bestCandidate, false,
                                                       false, false}
                   : nextCandidate();
    auto const candidate = selection.candidateIndex;
    m_lastCandidate = candidate;
    LaunchCall<Queue, FrameSpec, Bundle> call{.tuner = this,
                                              .queue = &queue,
                                              .frameSpec = &frameSpec,
                                              .prototype = &prototype,
                                              .runtimeSeconds = std::nullopt};
    auto const key = compileVariantKey(indicesFor(candidate));
    m_compileVariants.at(m_compileVariantIndices.at(key))(
        &call, candidate, selection.measure, selection.beginActivation,
        selection.endActivation);

    if (m_defaults.mode == TuningMode::onlineFixed && !m_complete &&
        m_queue->empty() &&
        m_scheduledCount + m_rejectedCount == m_candidateCount)
      finishTuning();

    auto const tunerInfo = info();
    return LaunchObservation{
        .candidateIndex = candidate,
        .configuration = normalizedConfiguration(candidate),
        .runtimeSeconds = call.runtimeSeconds,
        .recommendationSeconds = m_recommendationSecondsSinceLastLaunch,
        .measured = call.runtimeSeconds.has_value(),
        .tuningComplete = m_complete,
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
  template <typename Bundle>
  void validatePrototype(Bundle const &prototype) const {
    std::vector<std::string_view> markers;
    alpaka::apply(
        [&markers](auto const &...arguments) {
          (collectPrototypeMarker<decltype(arguments)>(markers), ...);
        },
        prototype.m_args);
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
    auto rawCandidateCount = std::size_t{1u};
    for (auto const size : m_dimensionSizes) {
      if (size == 0u ||
          rawCandidateCount > std::numeric_limits<std::size_t>::max() / size)
        throw std::overflow_error{"The Cartesian tuning space is too large."};
      rawCandidateCount *= size;
    }
    m_candidateCount = rawCandidateCount;
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
                               std::size_t &offset, Entry const &) const {
    using Values = typename Entry::values_type;
    constexpr auto arity = detail::candidateDimensionCount<Values>;
    auto const kind = [] {
      if constexpr (detail::isReservedLaunchName<Entry::name>)
        return LearnedDimensionKind::launch;
      else if constexpr (detail::isRVals<Values>)
        return LearnedDimensionKind::runtime;
      else if constexpr (detail::isCVals<Values>)
        return LearnedDimensionKind::compileTime;
      else
        return LearnedDimensionKind::categorical;
    }();
    for (std::size_t component = 0u; component < arity; ++component) {
      auto dimension = LearnedDimensionDescriptor{
          .name = std::string{Entry::nameView()},
          .kind = kind,
          .cardinality = m_dimensionSizes.at(offset + component),
          .componentIndex = component,
          .vectorArity = arity,
          .concreteValues = {}};
      dimension.concreteValues.reserve(dimension.cardinality);
      for (std::size_t candidate = 0u; candidate < dimension.cardinality;
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
    auto const properties = m_device.getDeviceProperties();
    descriptor.deviceClass = properties.warpSize > 1u ? LearnedDeviceClass::gpu
                                                      : LearnedDeviceClass::cpu;
    descriptor.contextFeatures = {
        {"candidate_count_log1p",
         std::log1p(static_cast<float>(m_candidateCount))},
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
      descriptor.legalCandidates.resize(m_candidateCount);
      for (std::size_t candidate = 0u; candidate < m_candidateCount;
           ++candidate)
        descriptor.legalCandidates[candidate] =
            static_cast<std::uint8_t>(candidateAccepted(indicesFor(candidate)));
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
    std::array<std::size_t, dimensionCount> indices{};
    for (std::size_t position = dimensionCount; position > 0u; --position) {
      auto const dimension = position - 1u;
      indices[dimension] = candidate % m_dimensionSizes[dimension];
      candidate /= m_dimensionSizes[dimension];
    }
    return indices;
  }

  [[nodiscard]] auto indicesFor(std::size_t candidate) const
      -> std::array<std::size_t, dimensionCount> {
    if (candidate >= m_candidateCount)
      throw std::out_of_range{"The candidate index is outside this tuner."};
    return rawIndicesFor(candidate);
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
                        entry.values.values().at(indices[dimensionOffset])));
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
    auto rawCandidate = std::size_t{0u};
    for (std::size_t dimension = 0u; dimension < dimensionCount; ++dimension) {
      auto const size = m_dimensionSizes[dimension];
      auto const index =
          size == 1u ? 0u
                     : static_cast<std::size_t>(std::lround(
                           static_cast<double>(configuration[dimension]) *
                           static_cast<double>(size - 1u)));
      rawCandidate = rawCandidate * size + index;
    }
    return rawCandidate;
  }

  /** @brief Clamped progress through the adaptive schedule horizon. */
  [[nodiscard]] auto adaptiveProgress() const -> double {
    auto const progress =
        static_cast<double>(m_adaptiveHorizonExecutionCount) /
        static_cast<double>(m_defaults.horizon);
    return detail::adaptiveProgressWithActiveHistory(
        progress, m_loadedFromCache,
        m_defaults.horizonOffsetWithActiveHistory);
  }

  /** @brief Apply the single shared legality and mode-specific admission path.
   *
   * Unseen legal candidates are scheduled directly. Adaptive revisits must
   * pass the normalized sigmoid and relative-score gates. Fixed mode rejects
   * measured proposals because its records are single-lifecycle.
   */
  [[nodiscard]] auto admitCandidate(std::size_t candidate)
      -> RecommendationDisposition {
    if (m_scheduled.at(candidate)) {
      ++m_activeDuplicateRejectedCount;
      return RecommendationDisposition::activeDuplicate;
    }
    if (m_rejected.at(candidate)) {
      ++m_restrictionRejectedCount;
      return RecommendationDisposition::restrictionRejected;
    }
    if (!candidateAccepted(indicesFor(candidate))) {
      m_rejected.at(candidate) = true;
      ++m_rejectedCount;
      ++m_restrictionRejectedCount;
      return RecommendationDisposition::restrictionRejected;
    }

    auto &history = m_histories.at(candidate);
    auto const revisit = !history.empty();
    if (revisit && m_defaults.mode == TuningMode::onlineFixed) {
      ++m_revisitRejectedCount;
      return RecommendationDisposition::revisitRejected;
    }
    if (revisit && m_defaults.mode == TuningMode::onlineAdaptive) {
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
      history.reopen();
      ++m_revisitAcceptedCount;
    } else {
      ++m_unseenAcceptedCount;
    }
    m_scheduled.at(candidate) = true;
    ++m_scheduledCount;
    return RecommendationDisposition::scheduled;
  }

  /** @brief Translate public configuration into per-candidate history policy.
   */
  [[nodiscard]] auto runtimeHistoryOptions() const
      -> detail::RuntimeHistoryOptions {
    return {m_defaults.warmupRuns,
            m_defaults.minimumRunsPerCandidate,
            m_defaults.runsPerCandidate,
            m_defaults.ciCheckInterval,
            m_defaults.ciZScore,
            m_defaults.ciRelativeWidth,
            m_defaults.outlierMadScale,
            m_defaults.historyWindowSize,
            m_defaults.mode == TuningMode::onlineFixed};
  }

  /** @brief Whether this online-adaptive run reached its schedule horizon. */
  [[nodiscard]] auto adaptiveHorizonReached() const noexcept -> bool {
    return m_adaptiveHorizonExecutionCount >= m_defaults.horizon;
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
      if (history.empty())
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
    auto const runtime = m_histories.at(candidate).statistics().estimate();
    if (runtime >= m_bestRetiredRuntime)
      return;
    m_bestRetiredRuntime = runtime;
    m_bestImprovements.push_back(detail::BestImprovement{
        .candidateIndex = candidate,
        .executionCount = m_executionCount,
        .retiredConfigurationCount = m_retiredConfigurationCount,
        .runtimeSeconds = runtime,
        .elapsedSeconds =
            std::chrono::duration<double>{std::chrono::steady_clock::now() -
                                          m_tuningStarted}
                .count()});
  }

  [[nodiscard]] auto
  runtimeForConfiguration(ParameterConfiguration const &configuration) const
      -> std::optional<RuntimeObservation> {
    try {
      auto const candidate = candidateForConfiguration(configuration);
      if (m_histories.empty() || m_histories.at(candidate).empty())
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

  private:
    Tuner const &m_tuner;
  };

  /** @brief Request and process exactly one strategy recommendation.
   *
   * Rejection returns no candidate; callers never synchronously retry the
   * strategy to replace the rejected point.
   */
  [[nodiscard]] auto recommendCandidate() -> std::optional<std::size_t> {
    if (m_defaults.mode == TuningMode::onlineFixed &&
        m_scheduledCount + m_rejectedCount == m_candidateCount)
      return std::nullopt;
    auto const start = std::chrono::steady_clock::now();
    auto const strategyContext = TunerStrategyView{*this};
    auto recommendation = m_strategy->recommend(strategyContext);
    validateParameterConfiguration(recommendation, std::span{m_dimensionSizes});
    auto const candidate = candidateForConfiguration(recommendation);
    auto const disposition = admitCandidate(candidate);
    m_strategy->recommendationResult(recommendation, disposition);
    m_recommendationSecondsSinceLastLaunch +=
        std::chrono::duration<double>{std::chrono::steady_clock::now() - start}
            .count();
    return disposition == RecommendationDisposition::scheduled
               ? std::optional<std::size_t>{candidate}
               : std::nullopt;
  }

  /** @brief Lazily bind persistence, histories, strategy, and active queue. */
  void initialiseScheduling() {
    if (m_queue || m_complete)
      return;
    m_tuningStarted = std::chrono::steady_clock::now();
    m_startedAtUnixSeconds =
        std::chrono::duration<double>{
            std::chrono::system_clock::now().time_since_epoch()}
            .count();
    m_histories.assign(m_candidateCount,
                       detail::RuntimeHistory{runtimeHistoryOptions()});
    m_scheduled.assign(m_candidateCount, false);
    m_rejected.assign(m_candidateCount, false);
    auto const loaded = loadCache();
    if (m_defaults.mode == TuningMode::offline) {
      if (!loaded)
        throw std::runtime_error{
            "Offline tuning requires a compatible history with at least one "
            "measured configuration."};
      return;
    }
    if (loaded && m_defaults.mode == TuningMode::onlineFixed && m_complete)
      return;
    if (m_defaults.strategy == StrategyKind::learnedHybrid) {
      auto const context = learnedModelContext();
      auto const options = LearnedHybridOptions{
          .candidatePoolSize = m_defaults.learnedCandidatePoolSize,
          .candidateBatchSize = m_defaults.learnedCandidateBatchSize};
      m_strategy =
          makeParameterStrategy(m_defaults.strategy, m_defaults.randomSeed,
                                &context, m_defaults.learnedModelFile, options);
      if (m_loadedLearnedAdapterState) {
        if (auto *learned =
                dynamic_cast<LearnedHybridStrategy *>(m_strategy.get()))
          static_cast<void>(learned->restoreResidualAdapterState(
              std::move(*m_loadedLearnedAdapterState)));
        m_loadedLearnedAdapterState.reset();
      }
    } else {
      m_strategy =
          makeParameterStrategy(m_defaults.strategy, m_defaults.randomSeed);
    }
    m_queue = std::make_unique<detail::CandidateQueue>(
        m_defaults.noiseCancellationWindow, m_defaults.maxConsecutiveRuns,
        false, m_random);
    refillQueue();
    if (m_queue->empty())
      throw std::invalid_argument{
          "The tuning-space restrictions rejected every candidate."};
  }

  /** @brief Make at most one recommendation attempt per vacant queue slot. */
  void refillQueue() {
    auto const attempts = m_defaults.noiseCancellationWindow - m_queue->size();
    for (std::size_t attempt = 0u; attempt < attempts && !m_queue->full();
         ++attempt) {
      auto const candidate = recommendCandidate();
      if (candidate)
        m_queue->insert(*candidate);
    }
  }

  /** @brief Select an admitted candidate or replay the current best.
   *
   * An empty queue after rejected refill attempts does not terminate adaptive
   * tuning. The best measured candidate is launched without measurement and a
   * new admission attempt occurs on the following application call.
   */
  [[nodiscard]] auto nextCandidate() -> detail::CandidateQueue::Selection {
    refillQueue();
    auto const selected = m_queue->next();
    if (selected)
      return *selected;
    if (auto const best = currentBestCandidate())
      return detail::CandidateQueue::Selection{*best, false, false, false};
    throw std::logic_error{
        "The tuning scheduler has neither an admitted nor a measured "
        "candidate."};
  }

  template <typename Queue, typename FrameSpec, typename Bundle>
  struct LaunchCall {
    Tuner *tuner;
    Queue const *queue;
    FrameSpec const *frameSpec;
    Bundle const *prototype;
    std::optional<double> runtimeSeconds;
  };

  using CompileVariantFunctor =
      std::function<void(void *, std::size_t, bool, bool, bool)>;

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
    auto const signature = detail::typeName<Bundle>();
    if (!m_compileVariants.empty()) {
      if (m_compileVariantSignature != signature)
        throw std::logic_error{"A tuner is bound to one KernelBundle type."};
      return;
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
      m_compileVariants.emplace_back([](void *rawCall, std::size_t candidate,
                                        bool measure, bool beginActivation,
                                        bool endActivation) {
        auto &call =
            *static_cast<LaunchCall<Queue, FrameSpec, Bundle> *>(rawCall);
        call.tuner->template launchCandidate<CompileValues...>(
            *call.queue, *call.frameSpec, *call.prototype, candidate, measure,
            beginActivation, endActivation, call.runtimeSeconds);
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
        return entry.values.values().at(indices[dimensionOffset]);
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
              prototype.m_kernelFn,
              rebuildArgument<std::remove_cvref_t<decltype(arguments)>,
                              CompileValues...>(arguments, indices)...};
        },
        prototype.m_args);
  }

  template <typename... CompileValues, typename Queue, typename LaunchSpec,
            typename Bundle>
  void launchCandidate(Queue const &queue, LaunchSpec const &prototypeLaunch,
                       Bundle const &prototype, std::size_t candidate,
                       bool measure, bool beginActivation, bool endActivation,
                       std::optional<double> &runtimeSeconds) {
    auto const indices = indicesFor(candidate);
    auto const bundle = rebuildBundle<CompileValues...>(prototype, indices);

    auto launch = [&] {
      if constexpr (alpaka::onHost::concepts::FrameSpec<
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
          queue.enqueue(thread, bundle);
        } else {
          queue.enqueue(frame, bundle);
        }
      } else if constexpr (alpaka::onHost::concepts::ThreadSpec<
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
        queue.enqueue(thread, bundle);
      } else {
        static_assert(alpaka::onHost::concepts::ThreadOrFrameSpec<
                          std::remove_cvref_t<LaunchSpec>>,
                      "alpakaTune::Tuner::enqueue requires "
                      "alpaka::onHost::FrameSpec or ThreadSpec.");
      }
    };

    ++m_executionCount;
    if (m_defaults.mode == TuningMode::onlineAdaptive)
      ++m_adaptiveHorizonExecutionCount;
    if (!measure) {
      launch();
      return;
    }
    auto &history = m_histories.at(candidate);
    if (beginActivation)
      history.beginActivation();
    auto const start = std::chrono::steady_clock::now();
    launch();
    alpaka::onHost::wait(queue);
    auto const elapsed = std::chrono::steady_clock::now() - start;
    runtimeSeconds = std::chrono::duration<double>{elapsed}.count();
    static_cast<void>(history.record(*runtimeSeconds));
    if (m_defaults.mode == TuningMode::onlineFixed)
      retireByMannWhitneyIfWarranted(candidate);
    if (m_defaults.mode == TuningMode::onlineAdaptive && endActivation)
      history.retire(detail::RuntimeCompletion::adaptiveVisit);
    if (history.isFinished()) {
      recordRetiredConfiguration(candidate);
      if (!m_queue->retire(candidate))
        throw std::logic_error{
            "The completed configuration was not active in the queue."};
      if (m_defaults.mode == TuningMode::onlineAdaptive) {
        m_scheduled.at(candidate) = false;
        --m_scheduledCount;
      }
    }
    if (m_defaults.mode == TuningMode::onlineFixed &&
        executionBudgetReached()) {
      finishTuningAtBudget(TunerCompletionReason::maximumExecutions);
      return;
    }
    if (m_defaults.mode == TuningMode::onlineFixed &&
        retiredConfigurationBudgetReached()) {
      finishTuningAtBudget(TunerCompletionReason::maximumRetiredConfigurations);
      return;
    }
    if (history.isFinished())
      refillQueue();
    stageCache(candidate, history.isFinished());
  }

  template <typename LaunchSpec, typename Bundle>
  [[nodiscard]] auto makeFingerprint(LaunchSpec const &launchSpec,
                                     Bundle const &prototype) const
      -> std::string {
    using BundleType = std::remove_cvref_t<Bundle>;
    std::ostringstream identity;
    identity << "schema=" << persistenceSchemaVersion << '\n';
    identity << "kernel=" << detail::typeName<typename BundleType::KernelFn>()
             << '\n';
    identity << "bundle=" << detail::typeName<BundleType>() << '\n';
    identity << "device=" << detail::identityName(m_device) << '\n';
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
    identity << "candidates=" << m_candidateCount << '\n';
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
    if constexpr (detail::isRVals<Values>) {
      for (auto const &value : entry.values.values())
        identity << detail::printable(value) << ',';
    } else {
      appendCValFingerprint<Values>(identity);
    }
    identity << ';';
  }

  /** Stable workload key excluding strategy, seed, budgets, and device. */
  [[nodiscard]] auto modelWorkloadId() const -> std::string {
    std::ostringstream identity;
    identity << "model-feature-schema="
             << LearnedModelContextDescriptor::featureSchemaVersion << '\n';
    identity << "kernel=" << m_kernelName << '\n';
    identity << "launch=" << m_launchSpecification << '\n';
    for (auto const &entry : m_identityEntries)
      identity << "context=" << entry << '\n';
    std::apply(
        [&identity](auto const &...entries) {
          (appendTunableFingerprint(identity, entries), ...);
        },
        m_tunables.entries());
    identity << "candidates=" << m_candidateCount << '\n';
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
    if (name == "all_configurations")
      return TunerCompletionReason::allConfigurations;
    if (name == "offline_replay")
      return TunerCompletionReason::offlineReplay;
    if (name == "maximum_executions")
      return TunerCompletionReason::maximumExecutions;
    if (name == "maximum_retired_configurations")
      return TunerCompletionReason::maximumRetiredConfigurations;
    return TunerCompletionReason::none;
  }

  [[nodiscard]] auto loadCache() -> bool {
#if ALPAKA_TUNE_HAS_JSON
    auto staged =
        m_persistence->stagedCache(persistenceSchemaVersion, m_fingerprint);
    auto store = nlohmann::json{};
    auto const *cachePointer = staged.get();
    if (cachePointer == nullptr) {
      if (!m_persistence->file || !m_persistence->readFile)
        return false;
      std::lock_guard lock{m_persistence->mutex};
      auto const &path = *m_persistence->file;
      if (!std::filesystem::exists(path))
        return false;
      std::ifstream input{path};
      if (!(input >> store) || !store.contains("contexts") ||
          !store["contexts"].is_object())
        throw std::runtime_error{
            "The alpakaTune persistent tuning file is invalid."};
      if (store.value("schema_version", 0) != persistenceSchemaVersion)
        throw std::runtime_error{
            "The alpakaTune persistent tuning file uses an incompatible "
            "schema; fresh histories are required."};
      if (!store["contexts"].contains(m_fingerprint))
        return false;
      cachePointer = &store["contexts"][m_fingerprint];
    }
    auto const &cache = *cachePointer;
    if (cache.value("fingerprint", "") != m_fingerprint)
      return false;
    if (cache.value("candidate_count", std::size_t{}) != m_candidateCount)
      return false;
    auto const completionReason = completionReasonFromName(
        cache.value("completion_reason", std::string{"none"}));
    auto const storedExecutionCount =
        cache.value("execution_count", std::size_t{});
    auto measuredCount = std::size_t{};
    try {
      auto const &storedHistories = cache.at("candidate_samples");
      if (!storedHistories.is_array() ||
          storedHistories.size() != m_candidateCount)
        return false;
      auto const rejected =
          cache.at("rejected_candidates").get<std::vector<bool>>();
      if (rejected.size() != m_candidateCount)
        return false;
      for (std::size_t candidate = 0u; candidate < m_candidateCount;
           ++candidate) {
        auto const samples =
            storedHistories.at(candidate).get<std::vector<double>>();
        if (rejected.at(candidate)) {
          if (!samples.empty())
            return false;
          continue;
        }
        if (samples.empty())
          continue;
        m_histories.at(candidate).restoreCompleted(samples);
        ++measuredCount;
      }
      if (measuredCount == 0u)
        return false;
      m_rejected = rejected;
      m_rejectedCount = static_cast<std::size_t>(
          std::count(m_rejected.begin(), m_rejected.end(), true));
      m_scheduled.assign(m_candidateCount, false);
      m_scheduledCount = 0u;
      if (m_defaults.mode == TuningMode::onlineFixed) {
        for (std::size_t candidate = 0u; candidate < m_candidateCount;
             ++candidate) {
          if (m_rejected.at(candidate) || m_histories.at(candidate).empty())
            continue;
          m_scheduled.at(candidate) = true;
          ++m_scheduledCount;
        }
      }

      m_retiredConfigurationCount =
          cache.value("retired_configuration_count", measuredCount);
      if (auto const admission = cache.find("admission");
          admission != cache.end() && admission->is_object()) {
        m_unseenAcceptedCount = admission->value("unseen_accepted", 0u);
        m_revisitAcceptedCount = admission->value("revisit_accepted", 0u);
        m_activeDuplicateRejectedCount =
            admission->value("active_duplicate_rejected", 0u);
        m_restrictionRejectedCount =
            admission->value("restriction_rejected", 0u);
        m_revisitRejectedCount = admission->value("revisit_rejected", 0u);
        m_scoreRejectedCount = admission->value("score_rejected", 0u);
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
    loadSerializedLearnedAdapter(cache);
    if (m_defaults.mode == TuningMode::offline) {
      m_complete = true;
      m_completionReason = TunerCompletionReason::offlineReplay;
    } else if (m_defaults.mode == TuningMode::onlineFixed &&
               completionReason != TunerCompletionReason::none) {
      m_complete = true;
      m_completionReason = completionReason;
      m_executionBudgetReached =
          completionReason == TunerCompletionReason::maximumExecutions;
    } else {
      m_complete = false;
      m_completionReason = TunerCompletionReason::none;
    }
    m_loadedFromCache = true;
    return true;
#else
    return false;
#endif
  }

  void finishTuning() {
    auto best = std::size_t{0u};
    auto bestSeconds = std::numeric_limits<double>::infinity();
    for (std::size_t candidate = 0u; candidate < m_candidateCount;
         ++candidate) {
      if (m_rejected.at(candidate))
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
    m_bestCandidate = best;
    m_complete = true;
    m_completionReason = TunerCompletionReason::allConfigurations;
    writeCache();
  }

  void finishTuningAtBudget(TunerCompletionReason reason) {
    if (reason != TunerCompletionReason::maximumExecutions &&
        reason != TunerCompletionReason::maximumRetiredConfigurations)
      throw std::logic_error{"Invalid completion-limit reason."};
    auto const best = currentBestCandidate();
    if (!best)
      throw std::logic_error{"The tuning completion limit was reached before "
                             "any candidate was measured."};
    m_bestCandidate = *best;
    m_complete = true;
    m_completionReason = reason;
    m_executionBudgetReached =
        reason == TunerCompletionReason::maximumExecutions;
    m_queue.reset();
    writeCache();
  }

  /** @brief Update the in-memory JSON snapshot after one measured launch.
   *
   * Sparse candidate arrays are updated in place. No filesystem operation is
   * performed; PersistenceStore flushes the newest snapshot at shutdown.
   */
  void stageCache(std::size_t candidate, bool schedulingChanged) {
#if ALPAKA_TUNE_HAS_JSON
    if (!m_stagedCache) {
      m_stagedCache = std::make_shared<nlohmann::json>(serializedCache());
    } else {
      auto &cache = *m_stagedCache;
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
          schedulingChanged)
        cache["learning"] = serializedLearningStatus();
      cache["admission"] = serializedAdmissionStatus();
      while (cache["best_improvements"].size() < m_bestImprovements.size())
        cache["best_improvements"].push_back(serializedImprovement(
            m_bestImprovements.at(cache["best_improvements"].size())));
    }
    m_persistence->stageCache(persistenceSchemaVersion, m_fingerprint,
                              m_stagedCache);
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
        {"runtime_seconds", improvement.runtimeSeconds},
        {"elapsed_seconds", improvement.elapsedSeconds},
        {"timestamp_unix_seconds",
         m_startedAtUnixSeconds + improvement.elapsedSeconds}};
  }

  [[nodiscard]] auto
  serializedCandidateConfiguration(std::size_t candidate) const
      -> nlohmann::json {
    auto configuration = nlohmann::json::object();
    auto const indices = indicesFor(candidate);
    std::apply(
        [this, &configuration, &indices](auto const &...entries) {
          (
              [&] {
                using Entry = std::remove_cvref_t<decltype(entries)>;
                static_cast<void>(withCandidateValue<Entry::name>(
                    indices, [&configuration](auto const &value) {
                      configuration[std::string{Entry::name.view()}] =
                          detail::printable(value);
                      return true;
                    }));
              }(),
              ...);
        },
        m_tunables.entries());
    return configuration;
  }

  [[nodiscard]] auto serializedLearnedModelContext() const -> nlohmann::json {
    auto const descriptor = learnedModelContext();
    auto result = nlohmann::json{
        {"feature_schema_version",
         LearnedModelContextDescriptor::featureSchemaVersion},
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

  void loadSerializedLearnedAdapter(nlohmann::json const &cache) {
    m_loadedLearnedAdapterState.reset();
    if (m_defaults.strategy != StrategyKind::learnedHybrid)
      return;
    try {
      auto const &learning = cache.at("learning");
      if (learning.value("model_digest", std::string{}) !=
          detail::fileFingerprint(m_defaults.learnedModelFile))
        return;
      auto const &adapter = learning.at("residual_adapter");
      if (adapter.value("state_version", 0) != 1)
        return;
      auto state = LearnedResidualAdapterState{
          .coefficients =
              adapter.at("coefficients").get<std::vector<double>>(),
          .observationsSinceUpdate =
              adapter.value("observations_since_update", std::size_t{}),
          .updateCount = adapter.value("update_count", std::size_t{})};
      for (auto const &observation : adapter.at("observations"))
        state.observations.push_back(
            {.rawIndex =
                 observation.at("raw_candidate_index").get<std::size_t>(),
             .features =
                 observation.at("features").get<std::vector<float>>(),
             .residual = observation.at("residual").get<double>()});
      m_loadedLearnedAdapterState = std::move(state);
    } catch (std::exception const &) {
      // Timing history remains useful when optional adapter state is absent or
      // malformed.
    }
  }

  [[nodiscard]] auto serializedLearningStatus() const -> nlohmann::json {
    auto result = nlohmann::json{
        {"requested_strategy", std::string{strategyName(m_defaults.strategy)}},
        {"model_file", m_defaults.learnedModelFile.string()},
        {"model_digest", detail::fileFingerprint(m_defaults.learnedModelFile)},
        {"fallback_strategy",
         std::string{strategyName(m_defaults.learnedFallback)}}};
    auto const *learned =
        dynamic_cast<LearnedHybridStrategy const *>(m_strategy.get());
    if (learned == nullptr)
      return result;
    result["status"] = learnedHybridStatusName(learned->status());
    result["artifact_load_status"] =
        learnedModelLoadStatusName(learned->artifactLoadStatus());
    result["status_message"] = learned->statusMessage();
    result["initialization_seconds"] = learned->initializationSeconds();
    result["cached_candidate_count"] = learned->cachedCandidateCount();
    result["candidate_pool_capacity"] = learned->candidatePoolCapacity();
    result["candidate_batch_size"] = learned->candidateBatchSize();
    result["peak_cached_candidate_count"] = learned->peakCachedCandidateCount();
    result["scored_candidate_count"] = learned->scoredCandidateCount();
    result["pool_refill_count"] = learned->poolRefillCount();
    result["candidate_stream_exhausted"] = learned->candidateStreamExhausted();
    result["incorporated_observation_count"] =
        learned->incorporatedObservationCount();
    result["adapter_update_count"] = learned->adapterUpdateCount();
    auto const state = learned->residualAdapterState();
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

  [[nodiscard]] auto serializedAdmissionStatus() const -> nlohmann::json {
    return {{"unseen_accepted", m_unseenAcceptedCount},
            {"revisit_accepted", m_revisitAcceptedCount},
            {"active_duplicate_rejected", m_activeDuplicateRejectedCount},
            {"restriction_rejected", m_restrictionRejectedCount},
            {"revisit_rejected", m_revisitRejectedCount},
            {"score_rejected", m_scoreRejectedCount}};
  }

  [[nodiscard]] auto serializedCache() const -> nlohmann::json {
    nlohmann::json cache;
    cache["fingerprint"] = m_fingerprint;
    cache["candidate_count"] = m_candidateCount;
    if (auto const best = currentBestCandidate())
      cache["best_candidate_index"] = *best;
    else
      cache["best_candidate_index"] = nullptr;
    cache["execution_count"] = m_executionCount;
    cache["adaptive_horizon_execution_count"] =
        m_adaptiveHorizonExecutionCount;
    cache["retired_configuration_count"] = m_retiredConfigurationCount;
    cache["execution_budget_reached"] = m_executionBudgetReached;
    cache["completion_reason"] = completionReasonName(m_completionReason);
    cache["started_at_unix_seconds"] = m_startedAtUnixSeconds;
    if (m_complete)
      cache["completed_at_unix_seconds"] =
          std::chrono::duration<double>{
              std::chrono::system_clock::now().time_since_epoch()}
              .count();
    cache["metadata"] = {
        {"identity_entries", m_identityEntries},
        {"kernel", m_kernelName},
        {"device", detail::identityName(m_device)},
        {"launch_specification", m_launchSpecification},
        {"mode", std::string{tuningModeName(m_defaults.mode)}},
        {"strategy", std::string{strategyName(m_defaults.strategy)}},
        {"model_context", serializedLearnedModelContext()}};
    cache["policy"] = {
        {"history_window_size", m_defaults.historyWindowSize},
        {"warmup_runs", m_defaults.warmupRuns},
        {"noise_cancellation_window", m_defaults.noiseCancellationWindow},
        {"max_consecutive_runs", m_defaults.maxConsecutiveRuns}};
    if (m_defaults.mode == TuningMode::onlineAdaptive) {
      cache["policy"]["horizon"] = m_defaults.horizon;
      cache["policy"]["revisit_admission_steepness"] =
          m_defaults.revisitAdmissionSteepness;
      cache["policy"]["score_temperature_start"] =
          m_defaults.scoreTemperatureStart;
      cache["policy"]["score_temperature_end"] =
          m_defaults.scoreTemperatureEnd;
      cache["policy"]["horizon_offset_with_active_history"] =
          m_defaults.horizonOffsetWithActiveHistory;
    }
    cache["admission"] = serializedAdmissionStatus();
    if (m_defaults.strategy == StrategyKind::learnedHybrid)
      cache["learning"] = serializedLearningStatus();
    if (m_defaults.mode == TuningMode::onlineFixed) {
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
    if (!m_stagedCache)
      m_stagedCache = std::make_shared<nlohmann::json>();
    *m_stagedCache = serializedCache();
    m_persistence->stageCache(persistenceSchemaVersion, m_fingerprint,
                              m_stagedCache);
#endif
  }

  TunerConfig m_defaults;
  std::shared_ptr<detail::PersistenceStore> m_persistence;
  TunablesType m_tunables;
  Device m_device;
  std::vector<std::string> m_identityEntries;
  std::vector<std::size_t> m_dimensionSizes;
  std::size_t m_candidateCount{1u};
  std::mt19937_64 m_random;
  std::uniform_real_distribution<double> m_admissionDistribution{0.0, 1.0};
  std::unique_ptr<ParameterStrategy> m_strategy;
  std::optional<LearnedResidualAdapterState> m_loadedLearnedAdapterState;
  std::unique_ptr<detail::CandidateQueue> m_queue;
  std::vector<bool> m_scheduled;
  std::vector<bool> m_rejected;
  std::size_t m_scheduledCount{};
  std::size_t m_rejectedCount{};
  std::size_t m_executionCount{};
  /** Executions in this adaptive process run; intentionally not restored. */
  std::size_t m_adaptiveHorizonExecutionCount{};
  std::size_t m_retiredConfigurationCount{};
  std::size_t m_unseenAcceptedCount{};
  std::size_t m_revisitAcceptedCount{};
  std::size_t m_activeDuplicateRejectedCount{};
  std::size_t m_restrictionRejectedCount{};
  std::size_t m_revisitRejectedCount{};
  std::size_t m_scoreRejectedCount{};
  double m_bestRetiredRuntime{std::numeric_limits<double>::infinity()};
  std::vector<detail::BestImprovement> m_bestImprovements;
  std::chrono::steady_clock::time_point m_tuningStarted{};
  double m_startedAtUnixSeconds{};
  std::vector<detail::RuntimeHistory> m_histories;
  std::vector<CompileVariantFunctor> m_compileVariants;
  std::unordered_map<std::string, std::size_t> m_compileVariantIndices;
  std::string m_compileVariantSignature;
  std::string m_fingerprint;
  std::string m_kernelName;
  std::string m_launchSpecification;
  std::size_t m_bestCandidate{std::numeric_limits<std::size_t>::max()};
  std::size_t m_lastCandidate{std::numeric_limits<std::size_t>::max()};
  double m_recommendationSecondsSinceLastLaunch{};
  TunerCompletionReason m_completionReason{TunerCompletionReason::none};
  bool m_complete{};
  bool m_loadedFromCache{};
  bool m_executionBudgetReached{};
#if ALPAKA_TUNE_HAS_JSON
  std::shared_ptr<nlohmann::json> m_stagedCache;
#endif
};

/** @brief Construct a device-bound tuner from an explicit policy.
 *
 * @param config Tuner policy, validated before construction.
 * @param tunables Runtime, compile-time, and launch-shape candidate bundle.
 * @param device Physical device to which subsequent queues must belong.
 * @param identityEntries Additional stable workload identity components.
 *
 * Identity-entry order does not affect the persistent fingerprint.
 */
template <typename TunablesType, typename Device, typename... IdentityEntries>
  requires detail::isTunableBundle<TunablesType>
[[nodiscard]] auto makeTuner(TunerConfig config, TunablesType tunables,
                             Device device,
                             IdentityEntries const &...identityEntries) {
  config.validate();
  std::vector<std::string> names{
      (detail::typeName<std::remove_cvref_t<IdentityEntries>>() + "=" +
       detail::identityName(identityEntries))...};
  std::sort(names.begin(), names.end());
  auto persistence =
      detail::persistenceStore(config.persistenceFile, config.persistenceRead,
                               config.persistenceWrite);
  return Tuner<TunablesType, Device>{std::move(config), std::move(persistence),
                                     std::move(tunables), std::move(device),
                                     std::move(names)};
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
