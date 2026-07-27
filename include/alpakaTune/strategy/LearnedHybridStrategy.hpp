// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "alpakaTune/interfaces/Strategy.hpp"
#include "alpakaTune/model/DeepSetsModel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace alpakaTune {

/** Whether learned inference is active or why deterministic fallback is used.
 */
enum class LearnedHybridStatus {
  active,                         ///< Native model inference is active.
  fallbackArtifactUnavailable,    ///< Artifact could not be loaded.
  fallbackArtifactIncompatible,   ///< Artifact metadata or inference failed.
  fallbackContextIncompatible,    ///< Runtime context violates model contract.
  fallbackCandidateSpaceTooLarge, ///< Legacy explicit size fallback.
};

/** @brief Return the stable diagnostic spelling of learned-model status. */
[[nodiscard]] constexpr auto
learnedHybridStatusName(LearnedHybridStatus status) noexcept
    -> std::string_view {
  switch (status) {
  case LearnedHybridStatus::active:
    return "active";
  case LearnedHybridStatus::fallbackArtifactUnavailable:
    return "fallback_artifact_unavailable";
  case LearnedHybridStatus::fallbackArtifactIncompatible:
    return "fallback_artifact_incompatible";
  case LearnedHybridStatus::fallbackContextIncompatible:
    return "fallback_context_incompatible";
  case LearnedHybridStatus::fallbackCandidateSpaceTooLarge:
    return "fallback_candidate_space_too_large";
  }
  return "unknown";
}

/** @brief Why learned hybrid chose its most recent raw proposal. */
enum class LearnedSelectionReason {
  predictedFast,        ///< Lowest adapted predicted log-runtime.
  uncertaintyDiversity, ///< Ensemble uncertainty plus embedding diversity.
  fallbackSpaceFilling, ///< Deterministic non-model candidate stream.
};

/** @brief Bounds learned inference and configures residual adaptation. */
struct LearnedHybridOptions {
  /** Four exploitation selections followed by one uncertainty/diversity pick.
   */
  std::size_t exploitationSelectionsPerCycle{4u};
  /** Total selections per exploitation/exploration cycle. */
  std::size_t selectionsPerCycle{5u};
  /** Newly retired observations required before refitting the adapter. */
  std::size_t adapterBatchSize{16u};
  /** L2 regularization for the small residual linear regression. */
  double ridgePenalty{1.0e-3};
  /** Weight of embedding distance in uncertainty/diversity selection. */
  double diversityWeight{1.0};
  /** Maximum candidates whose scores and embeddings remain resident. */
  std::size_t candidatePoolSize{4'096u};
  /** Maximum candidates evaluated by one native model call. */
  std::size_t candidateBatchSize{256u};
};

/** One measured residual retained by the lightweight learned adapter. */
struct LearnedResidualObservation {
  std::size_t rawIndex{};
  std::vector<float> features;
  double residual{};
};

/** Portable state required to resume the learned residual adapter. */
struct LearnedResidualAdapterState {
  std::vector<double> coefficients;
  std::vector<LearnedResidualObservation> observations;
  std::size_t observationsSinceUpdate{};
  std::size_t updateCount{};
};

/**
 * A frozen offline DeepSets ensemble with a small online residual adapter.
 *
 * Candidates enter a bounded, deterministically sampled pool and their model
 * predictions and embeddings are evaluated exactly once in batches. The model
 * chooses 80% predicted-fast points and 20% uncertainty/diversity points. A
 * ridge adapter is refit after each batch of retired RuntimeObservations and
 * only the active pool is re-sorted. When loading or compatibility fails, the
 * same object explicitly reports fallback status and traverses the same seeded
 * candidate stream without model scoring.
 */
class LearnedHybridStrategy final : public ParameterStrategy {
public:
  /** @brief Load a model artifact from disk and bind it to one context. */
  LearnedHybridStrategy(std::filesystem::path const &artifactPath,
                        LearnedModelContextDescriptor descriptor,
                        std::uint64_t seed, LearnedHybridOptions options = {})
      : LearnedHybridStrategy(loadLearnedModelArtifact(artifactPath),
                              std::move(descriptor), seed, options) {}

  /** @brief Bind an already loaded immutable artifact to one context. */
  LearnedHybridStrategy(std::shared_ptr<LearnedModelArtifact const> artifact,
                        LearnedModelContextDescriptor descriptor,
                        std::uint64_t seed, LearnedHybridOptions options = {})
      : LearnedHybridStrategy(
            LearnedModelLoadResult{
                .status = artifact ? LearnedModelLoadStatus::available
                                   : LearnedModelLoadStatus::invalidMetadata,
                .message =
                    artifact
                        ? std::string{}
                        : std::
                              string{"No learned-model artifact was supplied."},
                .artifact = std::move(artifact)},
            std::move(descriptor), seed, options) {}

  /** @brief Construct from an explicit artifact-load diagnostic. */
  LearnedHybridStrategy(LearnedModelLoadResult loadResult,
                        LearnedModelContextDescriptor descriptor,
                        std::uint64_t seed, LearnedHybridOptions options = {})
      : m_descriptor(std::move(descriptor)), m_options(options), m_random(seed),
        m_artifactLoadStatus(loadResult.status),
        m_statusMessage(std::move(loadResult.message)) {
    validateOptions();
    if (!loadResult) {
      m_status = LearnedHybridStatus::fallbackArtifactUnavailable;
      return;
    }
    try {
      m_model = std::make_unique<NativeDeepSetsModel>(
          std::move(loadResult.artifact), m_descriptor);
      m_status = LearnedHybridStatus::active;
    } catch (std::exception const &error) {
      m_status = LearnedHybridStatus::fallbackArtifactIncompatible;
      m_statusMessage = error.what();
    }
  }

  /** @brief Propose one pool candidate or one deterministic fallback point. */
  [[nodiscard]] auto recommend(StrategyContext const &context)
      -> ParameterConfiguration override {
    prepare(context);
    if (m_uncachedFallback)
      return randomConfiguration(context.parameterSizes().size());

    if (m_status == LearnedHybridStatus::active)
      reconcileObservations(context);

    replenishPool();
    if (m_candidates.empty())
      throw std::logic_error{"The learned strategy has no legal candidates."};

    auto position = std::size_t{};
    if (m_status == LearnedHybridStatus::active) {
      auto const explore = m_selectionCount % m_options.selectionsPerCycle >=
                           m_options.exploitationSelectionsPerCycle;
      position = explore ? selectExploration() : selectExploitation();
      m_lastSelectionReason = explore
                                  ? LearnedSelectionReason::uncertaintyDiversity
                                  : LearnedSelectionReason::predictedFast;
    } else {
      position = selectFallback();
      m_lastSelectionReason = LearnedSelectionReason::fallbackSpaceFilling;
    }

    auto &selected = m_candidates.at(position);
    m_lastRecommended = selected;
    ++m_selectionCount;
    return selected.configuration;
  }

  /** @brief Track only admitted model candidates for residual observation. */
  void recommendationResult(ParameterConfiguration const &,
                            RecommendationDisposition disposition) override {
    if (disposition == RecommendationDisposition::scheduled &&
        m_status == LearnedHybridStatus::active && m_lastRecommended)
      m_pendingCandidates.push_back(*m_lastRecommended);
    m_lastRecommended.reset();
  }

  /** @brief Whether native inference is active or which fallback is in use. */
  [[nodiscard]] auto status() const noexcept -> LearnedHybridStatus {
    return m_status;
  }
  /** @brief Result of parsing and validating the model artifact itself. */
  [[nodiscard]] auto artifactLoadStatus() const noexcept
      -> LearnedModelLoadStatus {
    return m_artifactLoadStatus;
  }
  /** @brief Human-readable artifact or context compatibility diagnostic. */
  [[nodiscard]] auto statusMessage() const noexcept -> std::string_view {
    return m_statusMessage;
  }
  /** @brief Reason assigned to the most recent raw proposal. */
  [[nodiscard]] auto lastSelectionReason() const noexcept
      -> LearnedSelectionReason {
    return m_lastSelectionReason;
  }
  /** @brief Candidates currently resident in the bounded learned pool. */
  [[nodiscard]] auto cachedCandidateCount() const noexcept -> std::size_t {
    return m_candidates.size();
  }
  /** @brief Configured upper bound for learned candidate metadata. */
  [[nodiscard]] auto candidatePoolCapacity() const noexcept -> std::size_t {
    return m_options.candidatePoolSize;
  }
  /** @brief Configured upper bound for one native scoring call. */
  [[nodiscard]] auto candidateBatchSize() const noexcept -> std::size_t {
    return m_options.candidateBatchSize;
  }
  /** @brief High-water mark of resident learned candidates. */
  [[nodiscard]] auto peakCachedCandidateCount() const noexcept -> std::size_t {
    return m_peakCachedCandidateCount;
  }
  /** @brief Total candidates evaluated by the frozen offline model. */
  [[nodiscard]] auto scoredCandidateCount() const noexcept -> std::size_t {
    return m_scoredCandidateCount;
  }
  /** @brief Number of bounded-pool population operations. */
  [[nodiscard]] auto poolRefillCount() const noexcept -> std::size_t {
    return m_poolRefillCount;
  }
  /** @brief Whether the deterministic candidate stream has no unseen points. */
  [[nodiscard]] auto candidateStreamExhausted() const noexcept -> bool {
    return m_candidateStreamExhausted;
  }
  /** @brief Measurements incorporated into residual regression so far. */
  [[nodiscard]] auto incorporatedObservationCount() const noexcept
      -> std::size_t {
    return m_observations.size();
  }
  /** @brief Number of completed residual-adapter fits. */
  [[nodiscard]] auto adapterUpdateCount() const noexcept -> std::size_t {
    return m_adapterUpdateCount;
  }
  /** @brief Snapshot coefficients and retained fit observations. */
  [[nodiscard]] auto residualAdapterState() const
      -> LearnedResidualAdapterState {
    auto state = LearnedResidualAdapterState{
        .coefficients = m_adapter,
        .observationsSinceUpdate = m_observationsSinceUpdate,
        .updateCount = m_adapterUpdateCount};
    state.observations.reserve(m_observations.size());
    for (auto const &observation : m_observations)
      state.observations.push_back({.rawIndex = observation.rawIndex,
                                    .features = observation.features,
                                    .residual = observation.residual});
    return state;
  }

  /** @brief Restore a compatible adapter before candidate-pool preparation.
   *
   * @return false when the active artifact or serialized dimensions do not
   * match. Invalid state is ignored without disabling the base model.
   */
  auto restoreResidualAdapterState(LearnedResidualAdapterState state) -> bool {
    if (m_initialised || m_status != LearnedHybridStatus::active || !m_model)
      return false;
    auto const featureCount = m_model->artifact().adapterFeatureCount();
    if ((!state.coefficients.empty() &&
         state.coefficients.size() != featureCount + 1u) ||
        state.observationsSinceUpdate >= m_options.adapterBatchSize ||
        !std::ranges::all_of(state.coefficients,
                             [](double value) { return std::isfinite(value); }))
      return false;
    auto const candidateCount = detail::candidateCount(m_descriptor.dimensions);
    auto indices = std::vector<std::size_t>{};
    indices.reserve(state.observations.size());
    for (auto const &observation : state.observations) {
      if (observation.rawIndex >= candidateCount ||
          observation.features.size() != featureCount ||
          !std::isfinite(observation.residual) ||
          !std::ranges::all_of(observation.features, [](float value) {
            return std::isfinite(value);
          }))
        return false;
      indices.push_back(observation.rawIndex);
    }
    std::ranges::sort(indices);
    if (std::ranges::adjacent_find(indices) != indices.end())
      return false;
    m_adapter = std::move(state.coefficients);
    m_observations.clear();
    m_observations.reserve(state.observations.size());
    for (auto &observation : state.observations)
      m_observations.push_back({.rawIndex = observation.rawIndex,
                                .features = std::move(observation.features),
                                .residual = observation.residual});
    m_observationsSinceUpdate = state.observationsSinceUpdate;
    m_adapterUpdateCount = state.updateCount;
    return true;
  }

  /** @brief Initialize the bounded candidate pool outside a timed recommend.
   *
   * This does not score the full Cartesian space. It evaluates at most the
   * configured bounded pool in configured-size batches.
   */
  void prepare(StrategyContext const &context) {
    if (m_initialised)
      return;
    auto const started = std::chrono::steady_clock::now();
    initialise(context);
    m_initializationSeconds =
        std::chrono::duration<double>{std::chrono::steady_clock::now() -
                                      started}
            .count();
  }

  /** @brief Whether model/context validation and initial pool scoring ran. */
  [[nodiscard]] auto prepared() const noexcept -> bool { return m_initialised; }
  /** @brief Wall time consumed by one-time preparation. */
  [[nodiscard]] auto initializationSeconds() const noexcept -> double {
    return m_initializationSeconds;
  }

private:
  struct Candidate {
    std::size_t rawIndex{};
    ParameterConfiguration configuration;
    double baseLogRuntime{};
    double uncertainty{};
    std::vector<float> adapterFeatures;
  };

  struct ResidualObservation {
    std::size_t rawIndex{};
    std::vector<float> features;
    double residual{};
  };

  void validateOptions() const {
    if (m_options.selectionsPerCycle == 0u ||
        m_options.exploitationSelectionsPerCycle >=
            m_options.selectionsPerCycle)
      throw std::invalid_argument{"The learned exploitation cycle is invalid."};
    if (m_options.adapterBatchSize == 0u ||
        !std::isfinite(m_options.ridgePenalty) ||
        m_options.ridgePenalty <= 0.0 ||
        !std::isfinite(m_options.diversityWeight) ||
        m_options.diversityWeight < 0.0 || m_options.candidatePoolSize == 0u ||
        m_options.candidateBatchSize == 0u ||
        m_options.candidateBatchSize > m_options.candidatePoolSize)
      throw std::invalid_argument{"The learned-hybrid options are invalid."};
  }

  [[nodiscard]] auto
  contextMatches(StrategyContext const &context) const noexcept -> bool {
    auto const actual = context.parameterSizes();
    if (actual.size() != m_descriptor.dimensions.size())
      return false;
    for (std::size_t index = 0u; index < actual.size(); ++index)
      if (actual[index] != m_descriptor.dimensions[index].cardinality)
        return false;
    return true;
  }

  void initialise(StrategyContext const &context) {
    m_initialised = true;
    auto const descriptorError = detail::validateLearnedContext(m_descriptor);
    if (!descriptorError.empty() || !contextMatches(context)) {
      m_status = LearnedHybridStatus::fallbackContextIncompatible;
      m_statusMessage =
          descriptorError.empty()
              ? "The learned descriptor and StrategyContext dimensions differ."
              : descriptorError;
      m_model.reset();
      m_uncachedFallback = true;
      return;
    }

    m_totalCandidateCount = detail::candidateCount(m_descriptor.dimensions);
    initialiseCandidateStream();
    m_candidates.reserve(
        std::min(m_totalCandidateCount, m_options.candidatePoolSize));
    replenishPool();
  }

  /** @brief Build a seeded, full-period permutation of Cartesian indices.
   *
   * A stride coprime to the candidate count visits every raw index exactly
   * once without allocating a full-space permutation.
   */
  void initialiseCandidateStream() {
    if (m_totalCandidateCount == 0u) {
      m_candidateStreamExhausted = true;
      return;
    }
    std::uniform_int_distribution<std::size_t> start(0u, m_totalCandidateCount -
                                                             1u);
    m_nextRawCandidate = start(m_random);
    if (m_totalCandidateCount == 1u) {
      m_candidateStride = 0u;
      return;
    }
    std::uniform_int_distribution<std::size_t> stride(
        1u, m_totalCandidateCount - 1u);
    m_candidateStride = stride(m_random);
    while (std::gcd(m_candidateStride, m_totalCandidateCount) != 1u) {
      ++m_candidateStride;
      if (m_candidateStride == m_totalCandidateCount)
        m_candidateStride = 1u;
    }
  }

  /** @brief Consume one index from the deterministic full-period stream. */
  [[nodiscard]] auto nextRawCandidate() -> std::optional<std::size_t> {
    if (m_candidateStreamExhausted)
      return std::nullopt;
    auto const result = m_nextRawCandidate;
    ++m_streamVisitedCount;
    if (m_streamVisitedCount == m_totalCandidateCount) {
      m_candidateStreamExhausted = true;
    } else if (m_nextRawCandidate >=
               m_totalCandidateCount - m_candidateStride) {
      m_nextRawCandidate -= m_totalCandidateCount - m_candidateStride;
    } else {
      m_nextRawCandidate += m_candidateStride;
    }
    return result;
  }

  /** @brief Score one bounded batch and append its metadata to the pool. */
  void appendCandidateBatch(std::vector<Candidate> candidates) {
    if (m_status == LearnedHybridStatus::active) {
      try {
        auto configurations = std::vector<ParameterConfiguration>{};
        configurations.reserve(candidates.size());
        for (auto const &candidate : candidates)
          configurations.push_back(candidate.configuration);
        auto predictions = m_model->predictBatch(
            std::span<ParameterConfiguration const>{configurations});
        auto const featureCount = m_model->artifact().adapterFeatureCount();
        for (std::size_t index = 0u; index < candidates.size(); ++index) {
          candidates[index].baseLogRuntime =
              predictions[index].logRuntimeSeconds;
          candidates[index].uncertainty = predictions[index].uncertainty;
          candidates[index].adapterFeatures.assign(
              predictions[index].embedding.begin(),
              predictions[index].embedding.begin() +
                  static_cast<std::ptrdiff_t>(featureCount));
        }
        m_scoredCandidateCount += candidates.size();
      } catch (std::exception const &error) {
        m_status = LearnedHybridStatus::fallbackArtifactIncompatible;
        m_statusMessage = error.what();
        m_model.reset();
        m_pendingCandidates.clear();
      }
    }
    std::ranges::move(candidates, std::back_inserter(m_candidates));
  }

  /** @brief Populate an empty bounded pool from the deterministic stream.
   *
   * The method never materializes the full Cartesian space. It applies the
   * persisted legality bitmap while filling at most candidatePoolSize entries.
   */
  void replenishPool() {
    if (!m_candidates.empty())
      return;
    auto appended = false;
    while (m_candidates.size() < m_options.candidatePoolSize &&
           !m_candidateStreamExhausted) {
      auto batch = std::vector<Candidate>{};
      batch.reserve(
          std::min(m_options.candidateBatchSize,
                   m_options.candidatePoolSize - m_candidates.size()));
      while (batch.size() < m_options.candidateBatchSize &&
             m_candidates.size() + batch.size() < m_options.candidatePoolSize) {
        auto const rawIndex = nextRawCandidate();
        if (!rawIndex)
          break;
        if (!m_descriptor.legalCandidates.empty() &&
            m_descriptor.legalCandidates[*rawIndex] == 0u)
          continue;
        batch.push_back(
            Candidate{.rawIndex = *rawIndex,
                      .configuration = detail::configurationForCandidate(
                          *rawIndex, m_descriptor.dimensions)});
      }
      if (batch.empty())
        break;
      appendCandidateBatch(std::move(batch));
      appended = true;
    }
    if (appended) {
      ++m_poolRefillCount;
      m_peakCachedCandidateCount =
          std::max(m_peakCachedCandidateCount, m_candidates.size());
    }
    if (!appended)
      return;
    if (m_status == LearnedHybridStatus::active) {
      rebuildExploitationOrder();
      buildExplorationOrder();
    }
  }

  [[nodiscard]] auto randomConfiguration(std::size_t dimensions)
      -> ParameterConfiguration {
    auto result = ParameterConfiguration(dimensions);
    for (auto &value : result)
      value = m_uniform(m_random);
    m_lastSelectionReason = LearnedSelectionReason::fallbackSpaceFilling;
    ++m_selectionCount;
    return result;
  }

  /** @brief Incorporate finished admitted proposals into residual training.
   *
   * Revisited raw candidates replace their previous residual observation so
   * the adapter follows the tuner's rolling timing window instead of growing
   * an unbounded duplicate dataset.
   */
  void reconcileObservations(StrategyContext const &context) {
    auto newlyIncorporated = std::size_t{};
    auto stillPending = std::vector<Candidate>{};
    stillPending.reserve(m_pendingCandidates.size());
    for (auto &candidate : m_pendingCandidates) {
      auto const observation = context.runtimeFor(candidate.configuration);
      if (!observation || !observation->isFinished()) {
        stillPending.push_back(std::move(candidate));
        continue;
      }
      if (!std::isfinite(observation->seconds) || observation->seconds <= 0.0)
        continue;
      auto replacement =
          ResidualObservation{.rawIndex = candidate.rawIndex,
                              .features = candidate.adapterFeatures,
                              .residual = std::log(observation->seconds) -
                                          candidate.baseLogRuntime};
      auto const existing = std::ranges::find(
          m_observations, candidate.rawIndex, &ResidualObservation::rawIndex);
      if (existing == m_observations.end())
        m_observations.push_back(std::move(replacement));
      else
        *existing = std::move(replacement);
      ++newlyIncorporated;
    }
    m_pendingCandidates = std::move(stillPending);
    m_observationsSinceUpdate += newlyIncorporated;
    if (m_observationsSinceUpdate >= m_options.adapterBatchSize) {
      fitResidualAdapter();
      m_observationsSinceUpdate %= m_options.adapterBatchSize;
    }
  }

  /** @brief Frozen prediction plus the current fitted residual correction. */
  [[nodiscard]] auto adaptedLogRuntime(Candidate const &candidate) const
      -> double {
    auto prediction = candidate.baseLogRuntime;
    if (m_adapter.empty())
      return prediction;
    prediction += m_adapter.front();
    for (std::size_t index = 0u; index < candidate.adapterFeatures.size();
         ++index)
      prediction += m_adapter[index + 1u] *
                    static_cast<double>(candidate.adapterFeatures[index]);
    return prediction;
  }

  [[nodiscard]] auto selectExploitation() -> std::size_t {
    if (m_exploitationOrder.empty())
      return selectAnyActive();
    // Exploitation is deliberately allowed to recommend the current predicted
    // best again. Tuner-owned admission decides whether it may be remeasured.
    return m_exploitationOrder.front();
  }

  [[nodiscard]] auto selectExploration() -> std::size_t {
    if (m_explorationOrder.empty())
      return selectAnyActive();
    auto const position =
        m_explorationOrder[m_explorationCursor % m_explorationOrder.size()];
    ++m_explorationCursor;
    return position;
  }

  [[nodiscard]] auto selectFallback() -> std::size_t {
    return selectAnyActive();
  }

  [[nodiscard]] auto selectAnyActive() const -> std::size_t {
    if (m_candidates.empty())
      throw std::logic_error{"The learned strategy has no active candidate."};
    return 0u;
  }

  /** @brief Sort pool positions by adapted predicted log-runtime. */
  void rebuildExploitationOrder() {
    m_exploitationOrder.resize(m_candidates.size());
    std::iota(m_exploitationOrder.begin(), m_exploitationOrder.end(),
              std::size_t{0u});
    std::ranges::sort(
        m_exploitationOrder, [this](std::size_t left, std::size_t right) {
          auto const leftRuntime = adaptedLogRuntime(m_candidates[left]);
          auto const rightRuntime = adaptedLogRuntime(m_candidates[right]);
          return leftRuntime < rightRuntime ||
                 (leftRuntime == rightRuntime &&
                  m_candidates[left].rawIndex < m_candidates[right].rawIndex);
        });
    m_exploitationCursor = 0u;
  }

  /** @brief Interleave uncertain candidates across embedding sign buckets. */
  void buildExplorationOrder() {
    auto const featureCount =
        m_candidates.empty() || m_options.diversityWeight == 0.0
            ? 0u
            : std::min<std::size_t>(
                  8u, m_candidates.front().adapterFeatures.size());
    auto means = std::vector<double>(featureCount, 0.0);
    for (auto const &candidate : m_candidates)
      for (std::size_t feature = 0u; feature < featureCount; ++feature)
        means[feature] +=
            static_cast<double>(candidate.adapterFeatures[feature]) /
            static_cast<double>(m_candidates.size());
    auto const bucketCount = std::size_t{1u} << featureCount;
    auto buckets = std::vector<std::vector<std::size_t>>(bucketCount);
    for (std::size_t position = 0u; position < m_candidates.size();
         ++position) {
      auto bucket = std::size_t{};
      for (std::size_t feature = 0u; feature < featureCount; ++feature)
        if (static_cast<double>(
                m_candidates[position].adapterFeatures[feature]) >=
            means[feature])
          bucket |= std::size_t{1u} << feature;
      buckets[bucket].push_back(position);
    }
    for (auto &bucket : buckets)
      std::ranges::sort(bucket, [this](std::size_t left, std::size_t right) {
        auto const leftUncertainty = m_candidates[left].uncertainty;
        auto const rightUncertainty = m_candidates[right].uncertainty;
        return leftUncertainty > rightUncertainty ||
               (leftUncertainty == rightUncertainty &&
                m_candidates[left].rawIndex < m_candidates[right].rawIndex);
      });
    auto bucketOrder = std::vector<std::size_t>{};
    for (std::size_t bucket = 0u; bucket < buckets.size(); ++bucket)
      if (!buckets[bucket].empty())
        bucketOrder.push_back(bucket);
    std::ranges::sort(
        bucketOrder, [&buckets, this](std::size_t left, std::size_t right) {
          auto const leftUncertainty =
              m_candidates[buckets[left].front()].uncertainty;
          auto const rightUncertainty =
              m_candidates[buckets[right].front()].uncertainty;
          return leftUncertainty > rightUncertainty ||
                 (leftUncertainty == rightUncertainty && left < right);
        });
    auto cursors = std::vector<std::size_t>(buckets.size(), 0u);
    m_explorationOrder.clear();
    m_explorationOrder.reserve(m_candidates.size());
    while (m_explorationOrder.size() < m_candidates.size()) {
      for (auto const bucket : bucketOrder) {
        if (cursors[bucket] < buckets[bucket].size())
          m_explorationOrder.push_back(buckets[bucket][cursors[bucket]++]);
      }
    }
    m_explorationCursor = 0u;
  }

  [[nodiscard]] static auto solve(std::vector<double> matrix,
                                  std::vector<double> rightHandSide,
                                  std::size_t size)
      -> std::optional<std::vector<double>> {
    constexpr double minimumPivot = 1.0e-14;
    for (std::size_t pivot = 0u; pivot < size; ++pivot) {
      auto selected = pivot;
      for (std::size_t row = pivot + 1u; row < size; ++row)
        if (std::abs(matrix[row * size + pivot]) >
            std::abs(matrix[selected * size + pivot]))
          selected = row;
      if (std::abs(matrix[selected * size + pivot]) < minimumPivot)
        return std::nullopt;
      if (selected != pivot) {
        for (std::size_t column = pivot; column < size; ++column)
          std::swap(matrix[pivot * size + column],
                    matrix[selected * size + column]);
        std::swap(rightHandSide[pivot], rightHandSide[selected]);
      }
      auto const divisor = matrix[pivot * size + pivot];
      for (std::size_t column = pivot; column < size; ++column)
        matrix[pivot * size + column] /= divisor;
      rightHandSide[pivot] /= divisor;
      for (std::size_t row = 0u; row < size; ++row) {
        if (row == pivot)
          continue;
        auto const factor = matrix[row * size + pivot];
        for (std::size_t column = pivot; column < size; ++column)
          matrix[row * size + column] -= factor * matrix[pivot * size + column];
        rightHandSide[row] -= factor * rightHandSide[pivot];
      }
    }
    return rightHandSide;
  }

  /** @brief Fit ridge regression from embeddings to observed log residuals.
   *
   * The frozen offline model is never modified. A successful fit only changes
   * m_adapter and consequently reorders the bounded exploitation pool.
   */
  void fitResidualAdapter() {
    if (m_observations.empty())
      return;
    auto const featureCount = m_observations.front().features.size();
    auto const size = featureCount + 1u;
    auto matrix = std::vector<double>(size * size, 0.0);
    auto rightHandSide = std::vector<double>(size, 0.0);
    for (auto const &observation : m_observations) {
      auto row = std::vector<double>(size, 1.0);
      for (std::size_t feature = 0u; feature < featureCount; ++feature)
        row[feature + 1u] = observation.features[feature];
      for (std::size_t left = 0u; left < size; ++left) {
        rightHandSide[left] += row[left] * observation.residual;
        for (std::size_t right = 0u; right < size; ++right)
          matrix[left * size + right] += row[left] * row[right];
      }
    }
    for (std::size_t index = 1u; index < size; ++index)
      matrix[index * size + index] += m_options.ridgePenalty;
    // A tiny intercept penalty keeps the system invertible for identical rows.
    matrix.front() += m_options.ridgePenalty * 1.0e-6;
    if (auto fitted =
            solve(std::move(matrix), std::move(rightHandSide), size)) {
      m_adapter = std::move(*fitted);
      ++m_adapterUpdateCount;
      rebuildExploitationOrder();
    }
  }

  LearnedModelContextDescriptor m_descriptor;
  LearnedHybridOptions m_options;
  std::mt19937_64 m_random;
  std::uniform_real_distribution<float> m_uniform{0.0f, 1.0f};
  LearnedModelLoadStatus m_artifactLoadStatus{
      LearnedModelLoadStatus::invalidMetadata};
  LearnedHybridStatus m_status{
      LearnedHybridStatus::fallbackArtifactUnavailable};
  std::string m_statusMessage;
  std::unique_ptr<NativeDeepSetsModel> m_model;
  std::vector<Candidate> m_candidates;
  std::vector<std::size_t> m_exploitationOrder;
  std::size_t m_exploitationCursor{};
  std::vector<std::size_t> m_explorationOrder;
  std::size_t m_explorationCursor{};
  std::vector<Candidate> m_pendingCandidates;
  std::optional<Candidate> m_lastRecommended;
  std::vector<ResidualObservation> m_observations;
  std::vector<double> m_adapter;
  std::size_t m_observationsSinceUpdate{};
  std::size_t m_adapterUpdateCount{};
  std::size_t m_selectionCount{};
  std::size_t m_totalCandidateCount{};
  std::size_t m_streamVisitedCount{};
  std::size_t m_nextRawCandidate{};
  std::size_t m_candidateStride{};
  std::size_t m_peakCachedCandidateCount{};
  std::size_t m_scoredCandidateCount{};
  std::size_t m_poolRefillCount{};
  LearnedSelectionReason m_lastSelectionReason{
      LearnedSelectionReason::fallbackSpaceFilling};
  bool m_initialised{};
  bool m_uncachedFallback{};
  bool m_candidateStreamExhausted{};
  double m_initializationSeconds{};
};

} // namespace alpakaTune
