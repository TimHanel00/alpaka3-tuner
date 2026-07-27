// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "alpakaTune/interfaces/Strategy.hpp"
#include "alpakaTune/strategy/BayesianOptimizationStrategy.hpp"
#include "alpakaTune/strategy/LearnedHybridStrategy.hpp"
#include "alpakaTune/strategy/RandomStrategy.hpp"
#include "alpakaTune/strategy/SimulatedAnnealingStrategy.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>

namespace alpakaTune {

class ExhaustiveStrategy final : public ParameterStrategy {
public:
  [[nodiscard]] auto recommend(StrategyContext const &context)
      -> ParameterConfiguration override {
    auto const dimensions = context.parameterSizes();
    auto configuration = ParameterConfiguration(dimensions.size(), 0.0f);
    auto candidate = m_nextCandidate++;
    for (std::size_t position = dimensions.size(); position > 0u; --position) {
      auto const dimension = position - 1u;
      auto const size = dimensions[dimension];
      auto const index = candidate % size;
      candidate /= size;
      configuration[dimension] = size == 1u ? 0.0f
                                            : static_cast<float>(index) /
                                                  static_cast<float>(size - 1u);
    }
    return configuration;
  }

private:
  std::size_t m_nextCandidate{};
};

[[nodiscard]] inline auto makeParameterStrategy(
    StrategyKind kind, std::uint64_t seed,
    LearnedModelContextDescriptor const *learnedContext = nullptr,
    std::filesystem::path const &learnedModel = {},
    LearnedHybridOptions learnedOptions = {})
    -> std::unique_ptr<ParameterStrategy> {
  switch (kind) {
  case StrategyKind::exhaustive:
    return std::make_unique<ExhaustiveStrategy>();
  case StrategyKind::random:
    return std::make_unique<RandomStrategy>(seed);
  case StrategyKind::simulatedAnnealing:
    return std::make_unique<SimulatedAnnealingStrategy>(seed);
  case StrategyKind::bayesianOptimization:
    return std::make_unique<BayesianOptimizationStrategy>(seed);
  case StrategyKind::learnedHybrid:
    if (learnedContext == nullptr)
      throw std::invalid_argument{
          "The learned-hybrid strategy requires a Tuner-owned model context."};
    return std::make_unique<LearnedHybridStrategy>(
        learnedModel, *learnedContext, seed, learnedOptions);
  }
  throw std::logic_error{"Unknown alpakaTune strategy kind."};
}

} // namespace alpakaTune
