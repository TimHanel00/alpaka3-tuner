/* Copyright 2025 Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once
#include "alpakaTune/config/ConfigRecord.hpp"
#include "alpakaTune/config/kruskalCompare.hpp"
#include "alpakaTune/core/peripherals/EnvironmentState.hpp"
#include "alpakaTune/interfaces/environmentVars.hpp"
#include "alpakaTune/interfaces/metricInterface.hpp"

namespace aTune::internal::config {
/**
 * @brief Prematurely retire underperforming configurations using a
 * Kruskal–Wallis test.
 *
 * Performs a non-parametric statistical comparison between the currently best
 * configuration and the given @p stored record. If the best configuration is
 * significantly better (p < 0.05), the inferior record is marked as @c Retired.
 * This avoids wasting evaluation time on configurations that are statistically
 * unlikely to outperform the current best.
 *
 * @tparam T_MetricInterface Metric interface used to compare configuration
 * quality.
 * @tparam T_Config          Configuration type (e.g. Config<...> or
 * NormalizedConfig<...>).
 *
 * @param stored Reference to the configuration record being evaluated.
 * @param state  Current environment state containing the best configuration.
 *
 * @note Uses @ref kruskalCompare internally to determine statistical
 * significance.
 * @see updateMetrics
 */
template <typename T_MetricInterface, typename T_Config>
void prematureConfigSkip(
    aTune::config::ConfigRecord<T_Config> &stored,
    aTune::core::peripherals::EnvironmentState<T_Config> &state) {
  auto const &best = state.getBestConfig().value().get();
  if (best == stored)
    return;
  if (stored.state != config::ConfigState::InProcess)
    return;
  auto res = best.compare(stored); // kruskal wallis comparison

  switch (res) {
  case config::Comparison::Greater: {
    stored.state = config::ConfigState::Retired;
    // best is higher then stored
    auto &config = compareGetBest<T_MetricInterface>(best, stored);
    if (best == config) {
      stored.state = config::ConfigState::Retired;
    }
  }
  case config::Comparison::Less: {
    // we know best is less then stored
    auto &config = compareGetBest<T_MetricInterface>(best, stored);
    // this returns the config that is best according to the metric interface
    // two scenarios:
    // 1. stored gets returned even though its greater then best --> meaning
    // higher is better --> stored stays
    // 2. best gets returned confirming that it is better AND also significantly
    // less --> meaning lower is better -- stored gets dropped prematurely
    if (best == config) {
      stored.state = config::ConfigState::Retired;
    }
  }
  case config::Comparison::Inconclusive: {
    break;
  }
  default:
    break;
  }
}

/**
 * @brief Update a configuration record with a new measurement and manage its
 * lifecycle state.
 *
 * Inserts a new timing sample into the given @p stored record, updates
 * statistics, and performs local break-criteria checks to determine whether
 * further measurements are required.
 *
 * @tparam kruskalWallisSkip Compile-time flag to enable or disable
 * early-retirement logic.
 * @tparam T_MetricInterface  Metric interface used to select the best
 * configuration.
 * @tparam T_Config           Configuration type (e.g. Config<...> or
 * NormalizedConfig<...>).
 *
 * @param stored Reference to the configuration record being updated.
 * @param state  Environment state managing all active configurations.
 * @param metric Newly measured metric value (e.g. runtime in nanoseconds).
 *
 * @see prematureConfigSkip
 * @see ConfigRecord
 */
template <bool kruskalWallisSkip, typename T_MetricInterface, typename T_Config>
void updateMetrics(aTune::config::ConfigRecord<T_Config> &stored,
                   core::peripherals::EnvironmentState<T_Config> &state,
                   double_t const &metric) {
  bool retired = (stored.state == ConfigState::Retired);

  stored.pushMetric(metric);
  if (retired)
    return;
  /// update best config.
  state.template updateBestConfig<T_MetricInterface>(stored);
  if (state.maxRunsPerConfig &&
      (stored.nr_runs >= state.maxRunsPerConfig.value())) {
    stored.state = ConfigState::Retired;
    return;
  }
  // std::cout<<" has min Runs: "<<state.minRunsPerConfig.value()<<std::endl;
  //  lower precedence -> CI criteria (+ additional requirement if minRuns is
  //  set)
  if (stored.state == ConfigState::CICriteriaReached &&
      (!state.minRunsPerConfig ||
       stored.nr_runs >= state.minRunsPerConfig.value())) {
    stored.state = ConfigState::Retired;
    return;
  }
  // still not finished -> try prematureSkip (kruskal wallis test)
  if constexpr (kruskalWallisSkip) {
    if (state.getBestConfig().has_value())
      prematureConfigSkip<T_MetricInterface>(stored);
  }
}
} // namespace aTune::internal::config
