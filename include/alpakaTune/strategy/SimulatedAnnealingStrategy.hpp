// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "alpakaTune/interfaces/Strategy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace alpakaTune {

/** Perturbs an accepted configuration with a cooling simulated-annealing step.
 */
class SimulatedAnnealingStrategy final : public ParameterStrategy {
public:
  /** @brief Construct a reproducible annealing proposal stream. */
  explicit SimulatedAnnealingStrategy(std::uint64_t seed) : m_random(seed) {}

  /** @brief Reconcile finished admitted points, then perturb accepted state. */
  [[nodiscard]] auto recommend(StrategyContext const &context)
      -> ParameterConfiguration override {
    reconcile(context);
    auto const dimensions = context.parameterSizes();
    if (m_current.empty()) {
      auto initial = ParameterConfiguration(dimensions.size());
      for (auto &value : initial)
        value = m_uniform(m_random);
      return initial;
    }

    auto proposal = m_current;
    if (proposal.size() != dimensions.size())
      proposal.assign(dimensions.size(), 0.5f);
    auto const radius = std::max(0.02f, 0.35f * m_temperature);
    for (auto &value : proposal)
      value = std::clamp(value + radius * m_normal(m_random), 0.0f, 1.0f);
    return proposal;
  }

  /** @brief Add only tuner-admitted proposals to pending annealing state. */
  void recommendationResult(ParameterConfiguration const &configuration,
                            RecommendationDisposition disposition) override {
    if (disposition == RecommendationDisposition::scheduled)
      m_pending.push_back(configuration);
  }

private:
  /** @brief Incorporate finished admitted proposals into annealing state.
   *
   * Only configurations reported as scheduled enter m_pending. Rejected raw
   * proposals therefore cannot influence the accepted state or temperature.
   */
  void reconcile(StrategyContext const &context) {
    if (!m_current.empty()) {
      if (auto const runtime = context.runtimeFor(m_current);
          runtime && runtime->isFinished())
        m_currentObjective = runtime->seconds;
    }
    for (auto iterator = m_pending.begin(); iterator != m_pending.end();) {
      auto const objective = context.runtimeFor(*iterator);
      if (!objective || !objective->isFinished()) {
        ++iterator;
        continue;
      }
      if (m_current.empty()) {
        m_current = *iterator;
        m_currentObjective = objective->seconds;
      } else if (objective->comparisonToBest == RuntimeComparison::slower) {
        // A rank test already established that this proposal is slower than
        // the incumbent. Keep annealing state independent, but do not move
        // into a statistically dominated point.
      } else {
        auto const relativeIncrease =
            (objective->seconds - m_currentObjective) /
            std::max(m_currentObjective, 1.0e-12);
        auto const acceptWorse =
            std::exp(-std::max(relativeIncrease, 0.0) /
                     std::max(static_cast<double>(m_temperature), 1.0e-6));
        if (relativeIncrease <= 0.0 || m_uniform(m_random) < acceptWorse) {
          m_current = *iterator;
          m_currentObjective = objective->seconds;
        }
      }
      m_temperature = std::max(0.02f, m_temperature * 0.995f);
      iterator = m_pending.erase(iterator);
    }
  }

  std::mt19937_64 m_random;
  std::uniform_real_distribution<float> m_uniform{0.0f, 1.0f};
  std::normal_distribution<float> m_normal{0.0f, 1.0f};
  ParameterConfiguration m_current;
  std::vector<ParameterConfiguration> m_pending;
  double m_currentObjective{std::numeric_limits<double>::infinity()};
  float m_temperature{1.0f};
};

} // namespace alpakaTune
