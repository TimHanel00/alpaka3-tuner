// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <cmath>
#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace alpakaTune {

/** A normalized point in the complete runtime, compile-time, and launch space.
 */
using ParameterConfiguration = std::vector<float>;

/** Lifecycle of the per-configuration timing record. */
enum class ConfigurationState {
  unmeasured, ///< No current activation and no timed launch yet.
  warmingUp,  ///< Current activation is consuming untimed warm-ups.
  measuring,  ///< Current activation is retaining timed launches.
  retired,    ///< Current fixed record or adaptive visit has ended.
};

/** Result of a rank comparison against the current statistically best record.
 */
enum class RuntimeComparison {
  unavailable,  ///< A directional comparison cannot yet be formed.
  faster,       ///< Statistically faster than the current best comparator.
  slower,       ///< Statistically slower than the current best comparator.
  inconclusive, ///< Available samples do not establish a direction.
};

/**
 * A stable view of one configuration's timing record.
 *
 * `seconds` is the robust estimate used by the tuner. `sampleCount` always
 * counts all measured samples, while `acceptedSampleCount` excludes samples
 * rejected by the MAD outlier filter.
 */
struct RuntimeObservation {
  /** Robust runtime estimate used for strategy and winner decisions. */
  double seconds{};
  /** Raw retained sample count. */
  std::size_t sampleCount{};
  /** Samples remaining after MAD-based decision filtering. */
  std::size_t acceptedSampleCount{};
  /** Current timing-record lifecycle state. */
  ConfigurationState state{ConfigurationState::unmeasured};
  /** Directional rank comparison with the current best measured record. */
  RuntimeComparison comparisonToBest{RuntimeComparison::unavailable};
  /** Whether the configured relative confidence width was reached. */
  bool confidenceReached{};

  /** @brief Whether this fixed record or adaptive activation has ended. */
  [[nodiscard]] constexpr auto isFinished() const noexcept -> bool {
    return state == ConfigurationState::retired;
  }
};

/** @brief Built-in normalized-parameter proposal algorithms. */
enum class StrategyKind {
  exhaustive,           ///< Deterministic Cartesian traversal.
  random,               ///< Seeded uniform proposals.
  simulatedAnnealing,   ///< Seeded local proposals with cooling acceptance.
  bayesianOptimization, ///< Bounded RBF lower-confidence-bound search.
  learnedHybrid,        ///< Frozen model plus online residual ordering.
};

/** @brief Return the stable YAML spelling of a strategy kind. */
[[nodiscard]] constexpr auto strategyName(StrategyKind strategy)
    -> std::string_view {
  switch (strategy) {
  case StrategyKind::exhaustive:
    return "exhaustive";
  case StrategyKind::random:
    return "random";
  case StrategyKind::simulatedAnnealing:
    return "simulated_annealing";
  case StrategyKind::bayesianOptimization:
    return "bayesian_optimization";
  case StrategyKind::learnedHybrid:
    return "learned_hybrid";
  }
  return "unknown";
}

/** @brief Parse a YAML strategy spelling.
 * @throws std::runtime_error if @p name is unsupported.
 */
[[nodiscard]] inline auto strategyFromName(std::string_view name)
    -> StrategyKind {
  if (name == "exhaustive")
    return StrategyKind::exhaustive;
  if (name == "random")
    return StrategyKind::random;
  if (name == "simulated_annealing")
    return StrategyKind::simulatedAnnealing;
  if (name == "bayesian_optimization")
    return StrategyKind::bayesianOptimization;
  if (name == "learned_hybrid")
    return StrategyKind::learnedHybrid;
  throw std::runtime_error{
      "YAML tuning.strategy must be exhaustive, random, simulated_annealing, "
      "bayesian_optimization, or learned_hybrid."};
}

/** @brief Validate a strategy's normalized output against the tuning shape.
 * @throws std::invalid_argument for wrong dimensionality or values outside
 * the inclusive normalized range [0, 1].
 */
inline void
validateParameterConfiguration(ParameterConfiguration const &configuration,
                               std::span<std::size_t const> dimensions) {
  if (configuration.size() != dimensions.size())
    throw std::invalid_argument{
        "A strategy returned the wrong number of normalized parameters."};
  for (auto const value : configuration) {
    if (!std::isfinite(value) || value < 0.0f || value > 1.0f)
      throw std::invalid_argument{"Strategy parameters must be finite floats "
                                  "in the inclusive range [0, 1]."};
  }
}

/**
 * @brief Recommends normalized parameter configurations for a tuning context.
 *
 * A recommendation is a vector with one float in [0, 1] for every tuning
 * dimension. Tuner owns the conversion to concrete candidate values.
 */
class StrategyContext {
public:
  virtual ~StrategyContext() = default;

  /** Number of discrete values represented by every normalized parameter. */
  [[nodiscard]] virtual auto parameterSizes() const noexcept
      -> std::span<std::size_t const> = 0;

  /**
   * Stable runtime observation for a normalized configuration, if sampled.
   *
   * The observation reports whether the record is still warming up or being
   * measured. This keeps record state useful to strategies without giving
   * them ownership of tuner state or a mutable history.
   */
  [[nodiscard]] virtual auto
  runtimeFor(ParameterConfiguration const &configuration) const
      -> std::optional<RuntimeObservation> = 0;
};

/** @brief Tuner-owned result of applying legality and admission policy. */
enum class RecommendationDisposition {
  scheduled,           ///< Candidate entered the active queue.
  activeDuplicate,     ///< Accepted: candidate was already queue-resident.
  restrictionRejected, ///< Candidate violated a tuning-space restriction.
  revisitRejected,     ///< Adaptive revisit-probability gate rejected it.
  scoreRejected,       ///< Adaptive relative-runtime gate rejected it.
};

/** @brief Base interface for algorithms that propose normalized parameters.
 *
 * Strategies do not own duplicate suppression, legality, queue state, or
 * application lifetime. The tuner reports each proposal's disposition once.
 * Both scheduled and activeDuplicate are accepted recommendations.
 */
class ParameterStrategy {
public:
  virtual ~ParameterStrategy() = default;

  /** @brief Produce one normalized proposal without an internal retry loop. */
  [[nodiscard]] virtual auto recommend(StrategyContext const &context)
      -> ParameterConfiguration = 0;

  /** @brief Observe the tuner-owned result of the preceding recommendation.
   *
   * Called exactly once for every vector returned by recommend(). Strategies
   * may use this to track only proposals that actually entered the queue.
   */
  virtual void recommendationResult(ParameterConfiguration const &,
                                    RecommendationDisposition) {}
};

} // namespace alpakaTune
