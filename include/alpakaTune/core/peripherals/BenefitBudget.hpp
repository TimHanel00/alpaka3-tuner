// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "alpakaTune/core/TunerConfig.hpp"
#include "alpakaTune/store/RuntimeHistory.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace alpakaTune::detail {

/** Pure cost/benefit accounting, independent of queues and wall clocks. */
class BenefitBudget {
public:
  explicit BenefitBudget(BudgetConfig config)
      : policy(config), remaining(config.expectedLaunches) {}

  void launched() noexcept {
    ++launches;
    if (remaining != 0u)
      --remaining;
  }
  void establishBaseline(double seconds) noexcept {
    if (!baselineEstablished) {
      baseline = seconds;
      baselineEstablished = true;
    }
  }
  [[nodiscard]] auto ceiling() const noexcept -> double {
    // Convert separately to avoid overflow when the application revises work.
    return policy.maximumOverheadFraction * baseline *
           (static_cast<double>(launches) + static_cast<double>(remaining));
  }
  [[nodiscard]] auto affordable(double cost) const noexcept -> bool {
    return remaining > 0u && spent + cost <= ceiling();
  }
  [[nodiscard]] auto worthwhile(double gain, double cost,
                                std::uint64_t activationLaunches = 1u) noexcept
      -> bool {
    auto const future =
        remaining > activationLaunches ? remaining - activationLaunches : 0u;
    expectedSavings = static_cast<double>(future) * gain;
    return affordable(cost) &&
           expectedSavings > policy.paybackMultiplier * cost;
  }
  void charge(double cost) noexcept { spent += std::max(0.0, cost); }
  void observeCost(double cost) noexcept {
    costEstimate = std::max(cost, costEstimate * 0.8);
  }
  [[nodiscard]] auto unseenGain(double incumbent) const noexcept -> double {
    return incumbent * policy.initialExpectedImprovement /
           std::sqrt(1.0 + static_cast<double>(unimprovedProposals));
  }
  [[nodiscard]] static auto
  uncertainty(RuntimeStatistics const &sample) noexcept -> double {
    auto const n = static_cast<double>(sample.acceptedSampleCount);
    if (n < 3.0)
      return std::max(sample.median * 0.1, sample.standardDeviation);
    return std::max(sample.median * 0.01,
                    1.2533141373155 * sample.standardDeviation) /
           std::sqrt(n);
  }
  /** Approximate positive improvement under independent Gaussian median errors.
   * This is an economic heuristic, not a calibrated probability guarantee. */
  [[nodiscard]] static auto
  expectedGain(RuntimeStatistics const &incumbent,
               RuntimeStatistics const &candidate) noexcept -> double {
    auto const difference = incumbent.median - candidate.median;
    auto const sigma =
        std::hypot(uncertainty(incumbent), uncertainty(candidate));
    if (sigma == 0.0)
      return std::max(0.0, difference);
    auto const z = difference / sigma;
    return std::max(0.0, difference * 0.5 * std::erfc(-z / std::sqrt(2.0)) +
                             sigma * std::exp(-0.5 * z * z) /
                                 std::sqrt(2.0 * std::acos(-1.0)));
  }

  BudgetConfig policy;
  std::uint64_t remaining{};
  std::uint64_t launches{};
  std::uint64_t measurements{};
  std::uint64_t productionLaunches{};
  std::uint64_t unimprovedProposals{};
  double baseline{};
  bool baselineEstablished{};
  double spent{};
  double costEstimate{};
  double expectedSavings{};
};

} // namespace alpakaTune::detail
