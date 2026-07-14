// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "alpakaTune/DerivedThreadSpec.hpp"
#include "alpakaTune/Session.hpp"
#include "alpakaTune/StrategyFactory.hpp"
#include "alpakaTune/Tunables.hpp"
#include "alpakaTune/detail/CandidateQueue.hpp"
#include "alpakaTune/detail/RuntimeHistory.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <span>
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

template <typename T> [[nodiscard]] auto printable(T const &value) -> std::string {
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

template <typename T> [[nodiscard]] auto identityName(T const &value) -> std::string {
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

template <FixedString Name, typename Value> struct SelectedCompileValue {
  static constexpr auto name = Name;
  using value_type = Value;
};

template <FixedString Name, typename... Values> struct FindSelectedCompile;
template <FixedString Name> struct FindSelectedCompile<Name> {
  static_assert(Name.view().empty(), "The compile-time marker was not selected.");
};
template <FixedString Name, typename Value, typename... Remaining>
struct FindSelectedCompile<Name, SelectedCompileValue<Name, Value>, Remaining...> {
  using type = Value;
};
template <FixedString Name, FixedString OtherName, typename Value,
          typename... Remaining>
struct FindSelectedCompile<Name, SelectedCompileValue<OtherName, Value>, Remaining...> {
  using type = typename FindSelectedCompile<Name, Remaining...>::type;
};

template <typename Values, std::size_t Index = 0u, typename Callable>
void dispatchCVal(std::size_t selectedIndex, Callable &&callable) {
  if constexpr (Index == Values::size) {
    throw std::out_of_range{"A compile-time tunable candidate index is out of range."};
  } else {
    if (selectedIndex == Index) {
      using Selected = std::tuple_element_t<Index, typename Values::values>;
      std::invoke(std::forward<Callable>(callable), std::type_identity<Selected>{});
      return;
    }
    dispatchCVal<Values, Index + 1u>(selectedIndex, std::forward<Callable>(callable));
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

/** A device-bound tuning context constructed from one YAML session. */
template <typename TunablesType, typename Device> class Context {
public:
  using Traits = detail::TunablesTraits<TunablesType>;
  using Entries = typename Traits::entries_type;
  static constexpr std::size_t dimensionCount = std::tuple_size_v<Entries>;
  static constexpr int persistenceSchemaVersion = 4;

  Context(SessionDefaults defaults, TunablesType tunables, Device device,
          std::vector<std::string> identityEntries)
      : m_defaults(std::move(defaults)), m_tunables(std::move(tunables)),
        m_device(std::move(device)), m_identityEntries(std::move(identityEntries)),
        m_random(m_defaults.randomSeed) {
    initialiseDimensions();
  }

  [[nodiscard]] auto isTuningComplete() const noexcept -> bool { return m_complete; }
  [[nodiscard]] auto loadedFromCache() const noexcept -> bool { return m_loadedFromCache; }
  [[nodiscard]] auto strategyKind() const noexcept -> StrategyKind {
    return m_defaults.strategy;
  }
  [[nodiscard]] auto bestCandidateIndex() const -> std::size_t {
    if (!m_complete)
      throw std::logic_error{"The tuning context has not selected a winner yet."};
    return m_bestCandidate;
  }
  [[nodiscard]] auto bestConfiguration() const -> ParameterConfiguration {
    return candidateConfiguration(bestCandidateIndex());
  }
  [[nodiscard]] auto candidateConfiguration(std::size_t candidate) const
      -> ParameterConfiguration {
    if (candidate >= m_candidateCount)
      throw std::out_of_range{"The candidate index is outside this tuning context."};
    return normalizedConfiguration(candidate);
  }
  [[nodiscard]] auto lastCandidateIndex() const noexcept -> std::size_t {
    return m_lastCandidate;
  }
  [[nodiscard]] auto candidateRuntimeStatistics(std::size_t candidate) const
      -> detail::RuntimeStatistics {
    if (m_histories.empty())
      throw std::logic_error{"The tuning context has not launched a candidate yet."};
    return m_histories.at(candidate).statistics();
  }
  [[nodiscard]] auto candidateRuntimeSamples(std::size_t candidate) const
      -> std::span<double const> {
    if (m_histories.empty())
      throw std::logic_error{"The tuning context has not launched a candidate yet."};
    return m_histories.at(candidate).samples();
  }

  template <typename Queue, typename FrameSpec, typename Kernel, typename... Args>
  void tune(Queue const &queue, FrameSpec const &frameSpec,
            alpaka::KernelBundle<Kernel, Args...> const &prototype) {
    if (!(queue.getDevice() == m_device))
      throw std::invalid_argument{"The queue device does not match this tuning context's device."};
    validatePrototype(prototype);
    auto const fingerprint = makeFingerprint<FrameSpec, decltype(prototype)>(frameSpec, prototype);
    bindFingerprint(fingerprint);
    initialiseScheduling();

    using Bundle = std::remove_cvref_t<decltype(prototype)>;
    initialiseCompileVariants<Queue, FrameSpec, Bundle>();
    auto const selection = m_complete
                               ? detail::CandidateQueue::Selection{m_bestCandidate, false}
                               : nextCandidate();
    auto const candidate = selection.candidateIndex;
    m_lastCandidate = candidate;
    LaunchCall<Queue, FrameSpec, Bundle> call{this, &queue, &frameSpec,
                                               &prototype};
    auto const key = compileVariantKey(indicesFor(candidate));
    m_compileVariants.at(m_compileVariantIndices.at(key))(
        &call, candidate, !m_complete, selection.beginActivation);

    if (!m_complete && m_queue->empty() && m_scheduledCount == m_candidateCount)
      finishTuning();
  }

private:
  template <typename Bundle> void validatePrototype(Bundle const &prototype) const {
    std::vector<std::string_view> markers;
    alpaka::apply(
        [&markers](auto const &...arguments) {
          ([&] {
            using Argument = std::remove_cvref_t<decltype(arguments)>;
            if constexpr (detail::isMarkedTunable<Argument>)
              markers.push_back(detail::IsMarkedTunable<Argument>::name.view());
          }(), ...);
        },
        prototype.m_args);
    std::apply(
        [&markers](auto const &...entries) {
          ([&] {
            using Entry = std::remove_cvref_t<decltype(entries)>;
            auto const consumed = detail::isReservedLaunchName<Entry::name> ||
                                  std::find(markers.begin(), markers.end(),
                                            Entry::nameView()) != markers.end();
            if (!consumed)
              throw std::invalid_argument{
                  "Every declared tunable must occur in the KernelBundle or be a reserved launch tunable."};
          }(), ...);
        },
        m_tunables.entries());
  }

  template <FixedString Name, typename Arguments, std::size_t Index = 0u>
  static consteval auto containsMarker() -> bool {
    if constexpr (Index == std::tuple_size_v<Arguments>)
      return false;
    else {
      using Argument = std::tuple_element_t<Index, Arguments>;
      if constexpr (detail::isMarkedTunable<Argument> &&
                    detail::sameName<Name, detail::IsMarkedTunable<Argument>::name>)
        return true;
      return containsMarker<Name, Arguments, Index + 1u>();
    }
  }

  void initialiseDimensions() {
    m_dimensionSizes.reserve(dimensionCount);
    std::apply(
        [this](auto const &...entries) {
          ((m_dimensionSizes.push_back(entrySize(entries))), ...);
        },
        m_tunables.entries());
    m_candidateCount = 1u;
    for (auto const size : m_dimensionSizes) {
      if (size == 0u || m_candidateCount > std::numeric_limits<std::size_t>::max() / size)
        throw std::overflow_error{"The Cartesian tuning space is too large."};
      m_candidateCount *= size;
    }
  }

  template <typename Entry>
  static auto entrySize(Entry const &entry) -> std::size_t {
    using Values = typename Entry::values_type;
    if constexpr (detail::isRVals<Values>)
      return entry.values.size();
    else
      return Values::size;
  }

  [[nodiscard]] auto indicesFor(std::size_t candidate) const -> std::array<std::size_t, dimensionCount> {
    std::array<std::size_t, dimensionCount> indices{};
    for (std::size_t position = dimensionCount; position > 0u; --position) {
      auto const dimension = position - 1u;
      indices[dimension] = candidate % m_dimensionSizes[dimension];
      candidate /= m_dimensionSizes[dimension];
    }
    return indices;
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

  [[nodiscard]] auto candidateForConfiguration(
      ParameterConfiguration const &configuration) const -> std::size_t {
    validateParameterConfiguration(configuration, std::span{m_dimensionSizes});
    auto candidate = std::size_t{0u};
    for (std::size_t dimension = 0u; dimension < dimensionCount; ++dimension) {
      auto const size = m_dimensionSizes[dimension];
      auto const index = size == 1u
                             ? 0u
                             : static_cast<std::size_t>(std::lround(
                                   static_cast<double>(configuration[dimension]) *
                                   static_cast<double>(size - 1u)));
      candidate = candidate * size + index;
    }
    return candidate;
  }

  [[nodiscard]] auto nearestUnscheduledCandidate(
      ParameterConfiguration const &recommendation) const -> std::size_t {
    auto const selected = candidateForConfiguration(recommendation);
    if (!m_scheduled.at(selected))
      return selected;

    auto nearest = std::numeric_limits<std::size_t>::max();
    auto nearestDistance = std::numeric_limits<double>::infinity();
    for (std::size_t candidate = 0u; candidate < m_candidateCount; ++candidate) {
      if (m_scheduled.at(candidate))
        continue;
      auto const configuration = normalizedConfiguration(candidate);
      auto distance = 0.0;
      for (std::size_t dimension = 0u; dimension < dimensionCount; ++dimension) {
        auto const difference = static_cast<double>(recommendation[dimension]) -
                                static_cast<double>(configuration[dimension]);
        distance += difference * difference;
      }
      if (distance < nearestDistance) {
        nearest = candidate;
        nearestDistance = distance;
      }
    }
    if (nearest == std::numeric_limits<std::size_t>::max())
      throw std::logic_error{"The strategy requested a candidate after the tuning space was exhausted."};
    return nearest;
  }

  [[nodiscard]] auto runtimeHistoryOptions() const -> detail::RuntimeHistoryOptions {
    return {m_defaults.warmupRuns,
            m_defaults.minimumRunsPerCandidate,
            m_defaults.runsPerCandidate,
            m_defaults.ciCheckInterval,
            m_defaults.ciZScore,
            m_defaults.ciRelativeWidth,
            m_defaults.outlierMadScale};
  }

  [[nodiscard]] auto currentBestCandidate() const -> std::optional<std::size_t> {
    auto best = std::optional<std::size_t>{};
    auto bestEstimate = std::numeric_limits<double>::infinity();
    for (std::size_t candidate = 0u; candidate < m_histories.size(); ++candidate) {
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
                                    m_defaults.mannWhitneyAlpha) == RuntimeComparison::slower)
      history.retire(detail::RuntimeCompletion::mannWhitneyU);
  }

  [[nodiscard]] auto runtimeForConfiguration(ParameterConfiguration const &configuration) const
      -> std::optional<RuntimeObservation> {
    try {
      auto const candidate = candidateForConfiguration(configuration);
      if (m_histories.empty() || m_histories.at(candidate).empty())
        return std::nullopt;
      auto const &history = m_histories.at(candidate);
      auto const statistics = history.statistics();
      auto comparison = RuntimeComparison::unavailable;
      if (auto const incumbent = currentBestCandidate(); incumbent && *incumbent != candidate)
        comparison = detail::mannWhitneyUCompare(history, m_histories.at(*incumbent),
                                                  m_defaults.mannWhitneyMinimumSamples,
                                                  m_defaults.mannWhitneyAlpha);
      return RuntimeObservation{statistics.estimate(), statistics.sampleCount,
                                statistics.acceptedSampleCount, history.state(), comparison,
                                statistics.confidenceReached};
    } catch (std::invalid_argument const &) {
      return std::nullopt;
    }
  }

  class ContextStrategyView final : public StrategyContext {
  public:
    explicit ContextStrategyView(Context const &context) : m_context(context) {}

    [[nodiscard]] auto parameterSizes() const noexcept
        -> std::span<std::size_t const> override {
      return m_context.m_dimensionSizes;
    }

    [[nodiscard]] auto runtimeFor(
        ParameterConfiguration const &configuration) const
        -> std::optional<RuntimeObservation> override {
      return m_context.runtimeForConfiguration(configuration);
    }

  private:
    Context const &m_context;
  };

  [[nodiscard]] auto recommendCandidate() -> std::size_t {
    auto const strategyContext = ContextStrategyView{*this};
    auto recommendation = m_strategy->recommend(strategyContext);
    validateParameterConfiguration(recommendation, std::span{m_dimensionSizes});
    return nearestUnscheduledCandidate(recommendation);
  }

  void initialiseScheduling() {
    if (m_queue || m_complete)
      return;
    m_histories.assign(m_candidateCount, detail::RuntimeHistory{runtimeHistoryOptions()});
    if (loadCache())
      return;
    m_strategy = makeParameterStrategy(m_defaults.strategy, m_defaults.randomSeed);
    m_queue = std::make_unique<detail::CandidateQueue>(
        m_defaults.noiseCancellationWindow, m_defaults.maxConsecutiveRuns,
        false, m_random);
    m_scheduled.assign(m_candidateCount, false);
    refillQueue();
  }

  void refillQueue() {
    while (!m_queue->full() && m_scheduledCount < m_candidateCount) {
      auto const candidate = recommendCandidate();
      m_scheduled.at(candidate) = true;
      ++m_scheduledCount;
      m_queue->insert(candidate);
    }
  }

  [[nodiscard]] auto nextCandidate() -> detail::CandidateQueue::Selection {
    auto const selected = m_queue->next();
    if (!selected)
      throw std::logic_error{"The tuning scheduler completed without selecting a winner."};
    return *selected;
  }

  template <typename Queue, typename FrameSpec, typename Bundle> struct LaunchCall {
    Context *context;
    Queue const *queue;
    FrameSpec const *frameSpec;
    Bundle const *prototype;
  };

  using CompileVariantFunctor = std::function<void(void *, std::size_t, bool, bool)>;

  [[nodiscard]] auto compileVariantKey(
      std::array<std::size_t, dimensionCount> const &indices) const -> std::string {
    std::ostringstream key;
    appendCompileVariantKey(key, indices, std::make_index_sequence<dimensionCount>{});
    return key.str();
  }

  template <std::size_t... Index>
  void appendCompileVariantKey(std::ostringstream &key,
                               std::array<std::size_t, dimensionCount> const &indices,
                               std::index_sequence<Index...>) const {
    ((appendCompileVariantKeyPart<std::tuple_element_t<Index, Entries>>(key, indices[Index])), ...);
  }

  template <typename Entry>
  static void appendCompileVariantKeyPart(std::ostringstream &key, std::size_t index) {
    if constexpr (detail::isCVals<typename Entry::values_type>)
      key << Entry::nameView() << '=' << index << ';';
  }

  template <typename Queue, typename FrameSpec, typename Bundle>
  void initialiseCompileVariants() {
    auto const signature = detail::typeName<Bundle>();
    if (!m_compileVariants.empty()) {
      if (m_compileVariantSignature != signature)
        throw std::logic_error{"A context is bound to one KernelBundle type."};
      return;
    }
    std::array<std::size_t, dimensionCount> indices{};
    appendCompileVariants<Queue, FrameSpec, Bundle, 0u>(indices);
    m_compileVariantSignature = signature;
  }

  template <typename Queue, typename FrameSpec, typename Bundle,
            std::size_t EntryIndex, typename... CompileValues>
  void appendCompileVariants(std::array<std::size_t, dimensionCount> &indices) {
    if constexpr (EntryIndex == dimensionCount) {
      auto const key = compileVariantKey(indices);
      m_compileVariantIndices.emplace(key, m_compileVariants.size());
      m_compileVariants.emplace_back(
          [](void *rawCall, std::size_t candidate, bool measure, bool beginActivation) {
            auto &call = *static_cast<LaunchCall<Queue, FrameSpec, Bundle> *>(rawCall);
            call.context->template launchCandidate<CompileValues...>(
                *call.queue, *call.frameSpec, *call.prototype, candidate, measure,
                beginActivation);
          });
    } else {
      using Entry = std::tuple_element_t<EntryIndex, Entries>;
      using Values = typename Entry::values_type;
      if constexpr (detail::isRVals<Values>) {
        appendCompileVariants<Queue, FrameSpec, Bundle, EntryIndex + 1u,
                              CompileValues...>(indices);
      } else {
        appendCValVariants<Queue, FrameSpec, Bundle, EntryIndex, Values, 0u,
                           CompileValues...>(indices);
      }
    }
  }

  template <typename Queue, typename FrameSpec, typename Bundle,
            std::size_t EntryIndex, typename Values, std::size_t ValueIndex,
            typename... CompileValues>
  void appendCValVariants(std::array<std::size_t, dimensionCount> &indices) {
    if constexpr (ValueIndex < Values::size) {
      using Entry = std::tuple_element_t<EntryIndex, Entries>;
      using Value = std::tuple_element_t<ValueIndex, typename Values::values>;
      indices[EntryIndex] = ValueIndex;
      appendCompileVariants<Queue, FrameSpec, Bundle, EntryIndex + 1u,
                            CompileValues...,
                            detail::SelectedCompileValue<Entry::name, Value>>(indices);
      appendCValVariants<Queue, FrameSpec, Bundle, EntryIndex, Values,
                         ValueIndex + 1u, CompileValues...>(indices);
    }
  }

  template <FixedString Name, typename... CompileValues>
  [[nodiscard]] auto valueFor(std::array<std::size_t, dimensionCount> const &indices) const {
    using Entry = typename Traits::template entry<Name>;
    constexpr auto entryIndex = Traits::template index<Name>;
    auto const &entry = std::get<entryIndex>(m_tunables.entries());
    using Values = typename Entry::values_type;
    if constexpr (detail::isRVals<Values>) {
      return entry.values.values().at(indices[entryIndex]);
    } else {
      using Selected = typename detail::FindSelectedCompile<Name, CompileValues...>::type;
      return Selected{};
    }
  }

  template <typename Argument, typename... CompileValues>
  [[nodiscard]] auto rebuildArgument(Argument const &argument,
                                     std::array<std::size_t, dimensionCount> const &indices) const {
    if constexpr (detail::isMarkedTunable<Argument>)
      return valueFor<detail::IsMarkedTunable<Argument>::name, CompileValues...>(indices);
    else
      return argument;
  }

  template <typename... CompileValues, typename Bundle>
  [[nodiscard]] auto rebuildBundle(Bundle const &prototype,
                                   std::array<std::size_t, dimensionCount> const &indices) const {
    return alpaka::apply(
        [this, &prototype, &indices](auto const &...arguments) {
          return alpaka::KernelBundle{
              prototype.m_kernelFn,
              rebuildArgument<std::remove_cvref_t<decltype(arguments)>, CompileValues...>(arguments, indices)...};
        },
        prototype.m_args);
  }

  template <typename... CompileValues, typename Queue, typename FrameSpec, typename Bundle>
  void launchCandidate(Queue const &queue, FrameSpec const &prototypeFrame,
                       Bundle const &prototype, std::size_t candidate, bool measure,
                       bool beginActivation) {
    auto const indices = indicesFor(candidate);
    auto const bundle = rebuildBundle<CompileValues...>(prototype, indices);
    auto const numFrames = [&] {
      if constexpr (Traits::template has<detail::numFramesName>)
        return valueFor<detail::numFramesName, CompileValues...>(indices);
      else
        return prototypeFrame.getNumFrames();
    }();
    auto const frameExtent = [&] {
      if constexpr (Traits::template has<detail::frameExtentName>)
        return valueFor<detail::frameExtentName, CompileValues...>(indices);
      else
        return prototypeFrame.getFrameExtents();
    }();
    auto const frame = alpaka::onHost::FrameSpec{
        numFrames, frameExtent, std::remove_cvref_t<FrameSpec>::getExecutor()};

    auto launch = [&] {
      if constexpr (Traits::template has<detail::numThreadsName> ||
                    Traits::template has<detail::numBlocksName>) {
        auto const derived = deriveThreadSpec(m_device, frame, bundle);
        auto const blocks = [&] {
          if constexpr (Traits::template has<detail::numBlocksName>)
            return valueFor<detail::numBlocksName, CompileValues...>(indices);
          else
            return derived.getNumBlocks();
        }();
        auto const threads = [&] {
          if constexpr (Traits::template has<detail::numThreadsName>)
            return valueFor<detail::numThreadsName, CompileValues...>(indices);
          else
            return derived.getNumThreads();
        }();
        auto const thread = alpaka::onHost::ThreadSpec{
            blocks, threads, std::remove_cvref_t<decltype(derived)>::getExecutor()};
        queue.enqueue(thread, bundle);
      } else {
        queue.enqueue(frame, bundle);
      }
    };

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
    static_cast<void>(history.record(std::chrono::duration<double>{elapsed}.count()));
    retireByMannWhitneyIfWarranted(candidate);
    if (history.isFinished()) {
      if (!m_queue->retire(candidate))
        throw std::logic_error{"The completed configuration was not active in the queue."};
      refillQueue();
    }
  }

  template <typename FrameSpec, typename Bundle>
  [[nodiscard]] auto makeFingerprint(FrameSpec const &frameSpec,
                                     Bundle const &prototype) const -> std::string {
    using BundleType = std::remove_cvref_t<Bundle>;
    std::ostringstream identity;
    identity << "schema=" << persistenceSchemaVersion << '\n';
    identity << "kernel=" << detail::typeName<typename BundleType::KernelFn>() << '\n';
    identity << "bundle=" << detail::typeName<BundleType>() << '\n';
    identity << "device=" << detail::identityName(m_device) << '\n';
    identity << "frame=" << detail::printable(frameSpec.getNumFrames()) << ':'
             << detail::printable(frameSpec.getFrameExtents()) << '\n';
    identity << "runs=" << m_defaults.runsPerCandidate << ':' << m_defaults.warmupRuns << ':'
             << m_defaults.noiseCancellationWindow << ':' << m_defaults.maxConsecutiveRuns << '\n';
    identity << "statistics=" << m_defaults.minimumRunsPerCandidate << ':'
             << m_defaults.ciCheckInterval << ':' << m_defaults.ciZScore << ':'
             << m_defaults.ciRelativeWidth << ':' << m_defaults.outlierMadScale << ':'
             << m_defaults.mannWhitneyEarlyStop << ':'
             << m_defaults.mannWhitneyMinimumSamples << ':'
             << m_defaults.mannWhitneyAlpha << '\n';
    identity << "strategy=" << strategyName(m_defaults.strategy) << ':'
             << m_defaults.randomSeed << '\n';
    for (auto const &entry : m_identityEntries)
      identity << "context=" << entry << '\n';
    std::apply(
        [&identity](auto const &...entries) {
          (appendTunableFingerprint(identity, entries), ...);
        },
        m_tunables.entries());
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

  template <typename Values, std::size_t Index = 0u>
  static void appendCValFingerprint(std::ostringstream &identity) {
    if constexpr (Index < Values::size) {
      using Value = std::tuple_element_t<Index, typename Values::values>;
      identity << detail::printable(Value::value) << ',';
      appendCValFingerprint<Values, Index + 1u>(identity);
    }
  }

  void bindFingerprint(std::string const &fingerprint) {
    if (m_fingerprint.empty()) {
      m_fingerprint = fingerprint;
      m_cachePath = m_defaults.persistenceDirectory / (fingerprint + ".json");
    } else if (m_fingerprint != fingerprint) {
      throw std::logic_error{"A context is bound to one kernel and one launch configuration."};
    }
  }

  [[nodiscard]] auto loadCache() -> bool {
#if ALPAKA_TUNE_HAS_JSON
    if (!std::filesystem::exists(m_cachePath))
      return false;
    std::ifstream input{m_cachePath};
    nlohmann::json cache;
    if (!(input >> cache) || cache.value("schema_version", 0) != persistenceSchemaVersion ||
        cache.value("fingerprint", "") != m_fingerprint)
      return false;
    auto const best = cache.value("best_candidate_index", m_candidateCount);
    if (best >= m_candidateCount)
      return false;
    try {
      auto const &storedHistories = cache.at("candidate_samples");
      if (!storedHistories.is_array() || storedHistories.size() != m_candidateCount)
        return false;
      for (std::size_t candidate = 0u; candidate < m_candidateCount; ++candidate) {
        auto const samples = storedHistories.at(candidate).get<std::vector<double>>();
        m_histories.at(candidate).restoreCompleted(samples);
      }
    } catch (std::exception const &) {
      return false;
    }
    m_bestCandidate = best;
    m_complete = true;
    m_loadedFromCache = true;
    return true;
#else
    return false;
#endif
  }

  void finishTuning() {
    auto best = std::size_t{0u};
    auto bestSeconds = std::numeric_limits<double>::infinity();
    for (std::size_t candidate = 0u; candidate < m_candidateCount; ++candidate) {
      auto const statistics = m_histories.at(candidate).statistics();
      if (statistics.sampleCount == 0u)
        throw std::logic_error{"A candidate completed without a measured launch."};
      auto const seconds = statistics.estimate();
      if (seconds < bestSeconds) {
        bestSeconds = seconds;
        best = candidate;
      }
    }
    m_bestCandidate = best;
    m_complete = true;
    writeCache();
  }

  void writeCache() const {
#if ALPAKA_TUNE_HAS_JSON
    std::filesystem::create_directories(m_cachePath.parent_path());
    nlohmann::json cache;
    cache["schema_version"] = persistenceSchemaVersion;
    cache["fingerprint"] = m_fingerprint;
    cache["best_candidate_index"] = m_bestCandidate;
    cache["candidate_samples"] = nlohmann::json::array();
    for (std::size_t candidate = 0u; candidate < m_candidateCount; ++candidate) {
      auto samples = nlohmann::json::array();
      for (auto const seconds : m_histories.at(candidate).samples())
        samples.push_back(seconds);
      cache["candidate_samples"].push_back(std::move(samples));
    }
    auto temporary = m_cachePath;
    temporary += ".tmp";
    std::ofstream output{temporary};
    if (!output)
      throw std::runtime_error{"Unable to write the alpakaTune persistent context cache."};
    output << cache.dump(2) << '\n';
    output.close();
    std::filesystem::rename(temporary, m_cachePath);
#endif
  }

  SessionDefaults m_defaults;
  TunablesType m_tunables;
  Device m_device;
  std::vector<std::string> m_identityEntries;
  std::vector<std::size_t> m_dimensionSizes;
  std::size_t m_candidateCount{1u};
  std::mt19937_64 m_random;
  std::unique_ptr<ParameterStrategy> m_strategy;
  std::unique_ptr<detail::CandidateQueue> m_queue;
  std::vector<bool> m_scheduled;
  std::size_t m_scheduledCount{};
  std::vector<detail::RuntimeHistory> m_histories;
  std::vector<CompileVariantFunctor> m_compileVariants;
  std::unordered_map<std::string, std::size_t> m_compileVariantIndices;
  std::string m_compileVariantSignature;
  std::string m_fingerprint;
  std::filesystem::path m_cachePath;
  std::size_t m_bestCandidate{std::numeric_limits<std::size_t>::max()};
  std::size_t m_lastCandidate{std::numeric_limits<std::size_t>::max()};
  bool m_complete{};
  bool m_loadedFromCache{};
};

class ContextBuilder {
public:
  explicit ContextBuilder(Session value = session()) : m_session(std::move(value)) {}

  template <typename TunablesType, typename Device, typename... IdentityEntries>
  [[nodiscard]] auto createContextWith(TunablesType tunables, Device device,
                                       IdentityEntries const &...identityEntries) const {
    std::vector<std::string> names{
        (detail::typeName<std::remove_cvref_t<IdentityEntries>>() + "=" +
         detail::identityName(identityEntries))...};
    std::sort(names.begin(), names.end());
    return Context<TunablesType, Device>{m_session.defaults(), std::move(tunables),
                                         std::move(device), std::move(names)};
  }

private:
  Session m_session;
};

inline auto Session::contextBuilder() const -> ContextBuilder {
  return ContextBuilder{*this};
}

[[nodiscard]] inline auto contextBuilder() -> ContextBuilder { return ContextBuilder{}; }
[[nodiscard]] inline auto contextBuilder(std::filesystem::path configuration) -> ContextBuilder {
  return ContextBuilder{session(std::move(configuration))};
}

} // namespace alpakaTune
