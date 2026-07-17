// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "alpakaTune/interfaces/Strategy.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace alpakaTune {

/**
 * @brief Gaussian-process-inspired Bayesian optimizer with an RBF surrogate.
 *
 * It retains a bounded observation history and minimizes a lower confidence
 * bound over random normalized candidates. Tuner still prevents duplicate
 * discrete candidates and guarantees that every candidate is measured.
 */
class BayesianOptimizationStrategy final : public ParameterStrategy {
public:
  explicit BayesianOptimizationStrategy(std::uint64_t seed) : m_random(seed) {}

  [[nodiscard]] auto recommend(StrategyContext const &context)
      -> ParameterConfiguration override {
    auto const dimensions = context.parameterSizes();
    auto const observations = measuredObservations(context);
    if (observations.empty()) {
      auto proposal = randomConfiguration(dimensions.size());
      m_requested.push_back(proposal);
      return proposal;
    }

    auto const model = buildModel(observations);
    auto best = randomConfiguration(dimensions.size());
    auto bestScore = acquisition(best, observations, model);
    for (std::size_t sample = 1u; sample < acquisitionSamples; ++sample) {
      auto proposal = randomConfiguration(dimensions.size());
      auto const score = acquisition(proposal, observations, model);
      if (score < bestScore) {
        best = std::move(proposal);
        bestScore = score;
      }
    }
    m_requested.push_back(best);
    if (m_requested.size() > maximumObservations)
      m_requested.erase(m_requested.begin());
    return best;
  }

private:
  struct Observation {
    ParameterConfiguration configuration;
    double objective;
  };

  struct Model {
    std::vector<double> kernel;
    std::vector<double> alpha;
    std::size_t size{};
    double scale{1.0};
  };

  static constexpr std::size_t maximumObservations = 64u;
  static constexpr std::size_t acquisitionSamples = 128u;
  static constexpr double lengthScale = 0.35;
  static constexpr double noise = 1.0e-9;
  static constexpr double exploration = 0.35;

  [[nodiscard]] auto randomConfiguration(std::size_t dimensions)
      -> ParameterConfiguration {
    auto configuration = ParameterConfiguration(dimensions);
    for (auto &value : configuration)
      value = m_uniform(m_random);
    return configuration;
  }

  [[nodiscard]] auto measuredObservations(StrategyContext const &context) const
      -> std::vector<Observation> {
    auto observations = std::vector<Observation>{};
    observations.reserve(m_requested.size());
    for (auto const &configuration : m_requested) {
      if (auto const runtime = context.runtimeFor(configuration);
          runtime && runtime->isFinished())
        observations.push_back({configuration, runtime->seconds});
    }
    return observations;
  }

  [[nodiscard]] static auto kernel(ParameterConfiguration const &left,
                                   ParameterConfiguration const &right)
      -> double {
    auto squaredDistance = 0.0;
    for (std::size_t index = 0u; index < left.size(); ++index) {
      auto const difference = static_cast<double>(left[index] - right[index]);
      squaredDistance += difference * difference;
    }
    return std::exp(-squaredDistance / (2.0 * lengthScale * lengthScale));
  }

  [[nodiscard]] static auto solve(std::vector<double> matrix,
                                  std::vector<double> rightHandSide,
                                  std::size_t size) -> std::vector<double> {
    for (std::size_t pivot = 0u; pivot < size; ++pivot) {
      auto selected = pivot;
      for (std::size_t row = pivot + 1u; row < size; ++row) {
        if (std::abs(matrix[row * size + pivot]) >
            std::abs(matrix[selected * size + pivot]))
          selected = row;
      }
      if (std::abs(matrix[selected * size + pivot]) < noise)
        throw std::runtime_error{
            "Bayesian strategy received a singular kernel matrix."};
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

  [[nodiscard]] static auto
  buildModel(std::vector<Observation> const &observations) -> Model {
    auto const size = observations.size();
    auto matrix = std::vector<double>(size * size);
    auto objective = std::vector<double>(size);
    auto scale = 0.0;
    for (std::size_t row = 0u; row < size; ++row) {
      objective[row] = observations[row].objective;
      scale += std::abs(objective[row]);
      for (std::size_t column = 0u; column < size; ++column) {
        matrix[row * size + column] =
            kernel(observations[row].configuration,
                   observations[column].configuration) +
            (row == column ? noise : 0.0);
      }
    }
    return {matrix, solve(matrix, objective, size), size,
            std::max(scale / static_cast<double>(size), 1.0e-12)};
  }

  [[nodiscard]] auto acquisition(ParameterConfiguration const &configuration,
                                 std::vector<Observation> const &observations,
                                 Model const &model) const -> double {
    auto covariance = std::vector<double>(model.size);
    for (std::size_t index = 0u; index < model.size; ++index)
      covariance[index] =
          kernel(configuration, observations[index].configuration);
    auto mean = 0.0;
    for (std::size_t index = 0u; index < model.size; ++index)
      mean += covariance[index] * model.alpha[index];
    auto const solvedCovariance = solve(model.kernel, covariance, model.size);
    auto variance = 1.0;
    for (std::size_t index = 0u; index < model.size; ++index)
      variance -= covariance[index] * solvedCovariance[index];
    return mean -
           exploration * model.scale * std::sqrt(std::max(variance, 0.0));
  }

  std::mt19937_64 m_random;
  std::uniform_real_distribution<float> m_uniform{0.0f, 1.0f};
  std::vector<ParameterConfiguration> m_requested;
};

} // namespace alpakaTune
