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

/** A normalized point in the complete runtime, compile-time, and launch space. */
using ParameterConfiguration = std::vector<float>;

/** Lifecycle of the per-configuration timing record. */
enum class ConfigurationState {
  unmeasured,
  warmingUp,
  measuring,
  retired,
};

/** Result of a rank comparison against the current statistically best record. */
enum class RuntimeComparison {
  unavailable,
  faster,
  slower,
  inconclusive,
};

/**
 * A stable view of one configuration's timing record.
 *
 * `seconds` is the robust estimate used by the tuner. `sampleCount` always
 * counts all measured samples, while `acceptedSampleCount` excludes samples
 * rejected by the MAD outlier filter.
 */
struct RuntimeObservation {
  double seconds{};
  std::size_t sampleCount{};
  std::size_t acceptedSampleCount{};
  ConfigurationState state{ConfigurationState::unmeasured};
  RuntimeComparison comparisonToBest{RuntimeComparison::unavailable};
  bool confidenceReached{};

  [[nodiscard]] constexpr auto isFinished() const noexcept -> bool {
    return state == ConfigurationState::retired;
  }
};

enum class StrategyKind {
  exhaustive,
  random,
  simulatedAnnealing,
  bayesianOptimization,
};

[[nodiscard]] constexpr auto strategyName(StrategyKind strategy) -> std::string_view {
  switch (strategy) {
  case StrategyKind::exhaustive:
    return "exhaustive";
  case StrategyKind::random:
    return "random";
  case StrategyKind::simulatedAnnealing:
    return "simulated_annealing";
  case StrategyKind::bayesianOptimization:
    return "bayesian_optimization";
  }
  return "unknown";
}

[[nodiscard]] inline auto strategyFromName(std::string_view name) -> StrategyKind {
  if (name == "exhaustive")
    return StrategyKind::exhaustive;
  if (name == "random")
    return StrategyKind::random;
  if (name == "simulated_annealing")
    return StrategyKind::simulatedAnnealing;
  if (name == "bayesian_optimization")
    return StrategyKind::bayesianOptimization;
  throw std::runtime_error{
      "YAML tuning.strategy must be exhaustive, random, simulated_annealing, or bayesian_optimization."};
}

inline void validateParameterConfiguration(ParameterConfiguration const &configuration,
                                           std::span<std::size_t const> dimensions) {
  if (configuration.size() != dimensions.size())
    throw std::invalid_argument{"A strategy returned the wrong number of normalized parameters."};
  for (auto const value : configuration) {
    if (!std::isfinite(value) || value < 0.0f || value > 1.0f)
      throw std::invalid_argument{"Strategy parameters must be finite floats in the inclusive range [0, 1]."};
  }
}

/**
 * @brief Recommends normalized parameter configurations for a tuning context.
 *
 * A recommendation is a vector with one float in [0, 1] for every tuning
 * dimension. Context owns the conversion to concrete candidate values.
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
   * them ownership of Context state or a mutable history.
   */
  [[nodiscard]] virtual auto runtimeFor(
      ParameterConfiguration const &configuration) const
      -> std::optional<RuntimeObservation> = 0;
};

class ParameterStrategy {
public:
  virtual ~ParameterStrategy() = default;

  [[nodiscard]] virtual auto recommend(StrategyContext const &context)
      -> ParameterConfiguration = 0;
};

} // namespace alpakaTune
