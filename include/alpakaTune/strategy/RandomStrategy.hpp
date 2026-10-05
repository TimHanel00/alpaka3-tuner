// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "alpakaTune/interfaces/Strategy.hpp"

#include <cstdint>
#include <random>

namespace alpakaTune {

/** Uniformly samples a normalized parameter configuration. */
class RandomStrategy final : public ParameterStrategy {
public:
  explicit RandomStrategy(std::uint64_t seed) : m_random(seed) {}

  [[nodiscard]] auto recommend(StrategyContext const &context)
      -> ParameterConfiguration override {
    auto configuration =
        ParameterConfiguration(context.parameterSizes().size());
    for (auto &value : configuration)
      value = m_distribution(m_random);
    return configuration;
  }

  [[nodiscard]] auto supportsCandidateCatalog() const noexcept
      -> bool override {
    return true;
  }
  [[nodiscard]] auto recommendCandidate(StrategyContext const &context)
      -> std::optional<std::size_t> override {
    std::optional<std::size_t> selected;
    std::size_t available{};
    for (std::size_t id{}; id < context.candidateCount(); ++id)
      if (context.candidateAvailable(id) &&
          std::uniform_int_distribution<std::size_t>{0u, available++}(
              m_random) == 0u)
        selected = id;
    return selected;
  }

private:
  std::mt19937_64 m_random;
  std::uniform_real_distribution<float> m_distribution{0.0f, 1.0f};
};

} // namespace alpakaTune
