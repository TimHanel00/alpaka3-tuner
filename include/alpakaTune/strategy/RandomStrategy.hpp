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

private:
  std::mt19937_64 m_random;
  std::uniform_real_distribution<float> m_distribution{0.0f, 1.0f};
};

} // namespace alpakaTune
