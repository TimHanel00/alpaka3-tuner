// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "alpakaTune/interfaces/Strategy.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace alpakaTune::detail {

/** @brief Controls one configuration's warm-up and measurement lifecycle. */
struct RuntimeHistoryOptions {
  /** Untimed launches at the beginning of each activation. */
  std::size_t warmupRuns{};
  /** Earliest sample count at which CI retirement is allowed. */
  std::size_t minimumMeasuredRuns{1u};
  /** Hard fixed-mode sample cap. */
  std::size_t maximumMeasuredRuns{1u};
  /** Sample cadence for confidence-interval checks. */
  std::size_t ciCheckInterval{10u};
  /** Z score used by the non-parametric median interval. */
  double ciZScore{2.576}; // 99 % normal approximation, as in the legacy tuner.
  /** Relative interval width required for confidence retirement. */
  double ciRelativeWidth{0.05};
  /** Median-absolute-deviation threshold for decision outliers. */
  double outlierMadScale{3.5};
  /** Maximum newest raw samples retained as a rolling window. */
  std::size_t historyWindowSize{10u};
  /** Enable CI and maximum-sample retirement for fixed mode. */
  bool automaticRetirement{true};
};

/** Statistics retained for a concrete configuration's timing history. */
struct RuntimeStatistics {
  /** Raw retained sample count. */
  std::size_t sampleCount{};
  /** MAD-filtered sample count used for robust estimates. */
  std::size_t acceptedSampleCount{};
  /** Minimum across the raw rolling window. */
  double rawMinimum{std::numeric_limits<double>::infinity()};
  /** Maximum across the raw rolling window. */
  double rawMaximum{-std::numeric_limits<double>::infinity()};
  /** Arithmetic mean across the raw rolling window. */
  double rawMean{};
  /** Arithmetic mean after MAD filtering. */
  double mean{};
  /** Median after MAD filtering and the tuner decision estimate. */
  double median{};
  /** Sample standard deviation after MAD filtering. */
  double standardDeviation{};
  /** Lower order-statistic bound of the median confidence interval. */
  double confidenceLow{};
  /** Upper order-statistic bound of the median confidence interval. */
  double confidenceHigh{};
  /** Confidence width divided by the robust median magnitude. */
  double confidenceRelativeWidth{std::numeric_limits<double>::infinity()};
  /** Whether confidenceRelativeWidth satisfies the configured threshold. */
  bool confidenceReached{};

  /** Median of MAD-filtered samples: the runtime used for decisions. */
  [[nodiscard]] auto estimate() const noexcept -> double { return median; }
};

/** @brief Why a configuration record ended its current lifecycle. */
enum class RuntimeCompletion {
  none,               ///< Record is open or has not started.
  confidenceInterval, ///< Fixed record reached its CI criterion.
  maximumSamples,     ///< Fixed record reached its sample cap.
  mannWhitneyU,       ///< Fixed record was statistically slower.
  adaptiveVisit,      ///< One adaptive activation burst ended.
};

/**
 * Per-configuration timing record modelled after the legacy ConfigRecord.
 *
 * Raw samples are retained for auditability and persistence. A MAD filter is
 * applied only when computing the decision statistics, preventing occasional
 * host scheduling spikes from moving the median/mean used by strategies.
 */
class RuntimeHistory {
public:
  explicit RuntimeHistory(RuntimeHistoryOptions options = {})
      : m_options(std::move(options)) {
    validateOptions();
  }

  /** Start a new queue activation; each activation receives its own warm-up. */
  void beginActivation() {
    if (isFinished())
      return;
    m_warmupRemaining = m_options.warmupRuns;
    m_state = m_warmupRemaining == 0u ? ConfigurationState::measuring
                                      : ConfigurationState::warmingUp;
  }

  /**
   * Record a timed launch. Returns true when this record reached retirement.
   * Warm-up launches deliberately do not enter the statistical history.
   */
  [[nodiscard]] auto record(double seconds) -> bool {
    if (!std::isfinite(seconds) || seconds < 0.0)
      throw std::invalid_argument{
          "A runtime measurement must be finite and non-negative."};
    if (isFinished())
      return true;
    if (m_state == ConfigurationState::unmeasured)
      beginActivation();
    if (m_warmupRemaining > 0u) {
      --m_warmupRemaining;
      if (m_warmupRemaining == 0u)
        m_state = ConfigurationState::measuring;
      return false;
    }

    m_samples.push_back(seconds);
    if (m_samples.size() > m_options.historyWindowSize)
      m_samples.erase(m_samples.begin());
    ++m_currentRunSampleCount;
    rebuildAcceptedSamples();
    auto const current = statistics();
    if (!m_options.automaticRetirement)
      return false;
    if (m_currentRunSampleCount >= m_options.maximumMeasuredRuns) {
      retire(RuntimeCompletion::maximumSamples);
    } else if (m_currentRunSampleCount >= m_options.minimumMeasuredRuns &&
               m_currentRunSampleCount % m_options.ciCheckInterval == 0u &&
               current.confidenceReached) {
      retire(RuntimeCompletion::confidenceInterval);
    }
    return isFinished();
  }

  /** @brief End the current lifecycle with at least one retained sample. */
  void retire(RuntimeCompletion completion) {
    if (m_samples.empty())
      throw std::logic_error{
          "A configuration cannot retire without a measured runtime."};
    m_completion = completion;
    m_state = ConfigurationState::retired;
  }

  /** Make a completed record eligible for another adaptive queue residency. */
  void reopen() {
    m_completion = RuntimeCompletion::none;
    m_warmupRemaining = 0u;
    m_state = ConfigurationState::unmeasured;
  }

  /** @brief Start a new online run without discarding retained timings. */
  void resetForNewRun() {
    m_currentRunSampleCount = 0u;
    m_completion = RuntimeCompletion::none;
    m_warmupRemaining = 0u;
    m_state = ConfigurationState::unmeasured;
  }

  /** Restore a persisted, completed record without replaying warm-up state. */
  void restoreCompleted(std::span<double const> samples) {
    if (samples.empty())
      throw std::invalid_argument{
          "A persisted runtime history must contain a measurement."};
    auto const first = samples.size() > m_options.historyWindowSize
                           ? samples.end() - static_cast<std::ptrdiff_t>(
                                                 m_options.historyWindowSize)
                           : samples.begin();
    m_samples.assign(first, samples.end());
    for (auto const sample : m_samples) {
      if (!std::isfinite(sample) || sample < 0.0)
        throw std::invalid_argument{
            "A persisted runtime measurement is invalid."};
    }
    rebuildAcceptedSamples();
    m_currentRunSampleCount = 0u;
    m_warmupRemaining = 0u;
    m_completion = RuntimeCompletion::maximumSamples;
    m_state = ConfigurationState::retired;
  }

  /** @brief Restore a lossy compact summary as median-valued samples.
   *
   * The compact schema deliberately omits the raw distribution. Repeating the
   * persisted median preserves its decision estimate and retained measurement
   * weight while respecting the active rolling-window limit.
   */
  void restoreSummary(double median, std::size_t measurementCount) {
    if (!std::isfinite(median) || median < 0.0 || measurementCount == 0u)
      throw std::invalid_argument{
          "A compact runtime summary requires a finite non-negative median "
          "and a positive measurement count."};
    auto const retained =
        std::min(measurementCount, m_options.historyWindowSize);
    m_samples.assign(retained, median);
    rebuildAcceptedSamples();
    m_currentRunSampleCount = 0u;
    m_warmupRemaining = 0u;
    m_completion = RuntimeCompletion::maximumSamples;
    m_state = ConfigurationState::retired;
  }

  /** @brief Current warm-up, measurement, or retired state. */
  [[nodiscard]] auto state() const noexcept -> ConfigurationState {
    return m_state;
  }
  /** @brief Reason the current lifecycle ended, or none while open. */
  [[nodiscard]] auto completion() const noexcept -> RuntimeCompletion {
    return m_completion;
  }
  /** @brief Whether the current lifecycle is retired. */
  [[nodiscard]] auto isFinished() const noexcept -> bool {
    return m_state == ConfigurationState::retired;
  }
  /** @brief Whether no timing sample is retained. */
  [[nodiscard]] auto empty() const noexcept -> bool {
    return m_samples.empty();
  }
  /** @brief Retained timings collected since the latest online-run reset. */
  [[nodiscard]] auto currentRunSampleCount() const noexcept -> std::size_t {
    return m_currentRunSampleCount;
  }
  /** @brief Raw retained rolling window, including decision outliers. */
  [[nodiscard]] auto samples() const noexcept -> std::span<double const> {
    return m_samples;
  }
  /** @brief MAD-filtered samples used by estimates and comparisons. */
  [[nodiscard]] auto acceptedSamples() const noexcept
      -> std::span<double const> {
    return m_acceptedSamples;
  }

  /** @brief Recompute robust statistics for the current rolling window. */
  [[nodiscard]] auto statistics() const -> RuntimeStatistics {
    RuntimeStatistics result;
    result.sampleCount = m_samples.size();
    result.acceptedSampleCount = m_acceptedSamples.size();
    if (m_samples.empty())
      return result;

    for (auto const sample : m_samples) {
      result.rawMinimum = std::min(result.rawMinimum, sample);
      result.rawMaximum = std::max(result.rawMaximum, sample);
      result.rawMean += sample;
    }
    result.rawMean /= static_cast<double>(m_samples.size());

    auto const &accepted =
        m_acceptedSamples.empty() ? m_samples : m_acceptedSamples;
    for (auto const sample : accepted)
      result.mean += sample;
    result.mean /= static_cast<double>(accepted.size());

    auto sorted = std::vector<double>{accepted.begin(), accepted.end()};
    std::sort(sorted.begin(), sorted.end());
    result.median = medianOfSorted(sorted);
    if (sorted.size() > 1u) {
      for (auto const sample : accepted) {
        auto const difference = sample - result.mean;
        result.standardDeviation += difference * difference;
      }
      result.standardDeviation = std::sqrt(
          result.standardDeviation / static_cast<double>(accepted.size() - 1u));
    }

    if (sorted.size() < 5u)
      return result;

    auto const lowerIndex = static_cast<std::size_t>(std::max(
        0.0, std::floor((static_cast<double>(sorted.size()) -
                         m_options.ciZScore *
                             std::sqrt(static_cast<double>(sorted.size()))) /
                        2.0)));
    auto const upperIndex = static_cast<std::size_t>(
        std::min(static_cast<double>(sorted.size() - 1u),
                 std::ceil((static_cast<double>(sorted.size()) +
                            m_options.ciZScore *
                                std::sqrt(static_cast<double>(sorted.size()))) /
                           2.0)));
    result.confidenceLow = sorted.at(lowerIndex);
    result.confidenceHigh = sorted.at(upperIndex);
    auto const scale =
        std::max(std::abs(result.median), std::numeric_limits<double>::min());
    result.confidenceRelativeWidth =
        (result.confidenceHigh - result.confidenceLow) / scale;
    result.confidenceReached =
        result.confidenceRelativeWidth <= m_options.ciRelativeWidth;
    return result;
  }

private:
  static auto medianOfSorted(std::vector<double> const &values) -> double {
    auto const middle = values.size() / 2u;
    if (values.size() % 2u == 0u)
      return (values.at(middle - 1u) + values.at(middle)) / 2.0;
    return values.at(middle);
  }

  void validateOptions() const {
    if (m_options.minimumMeasuredRuns == 0u ||
        m_options.maximumMeasuredRuns == 0u ||
        m_options.minimumMeasuredRuns > m_options.maximumMeasuredRuns)
      throw std::invalid_argument{
          "Runtime history requires 0 < minimum runs <= maximum runs."};
    if (m_options.historyWindowSize == 0u ||
        (m_options.automaticRetirement &&
         m_options.maximumMeasuredRuns > m_options.historyWindowSize))
      throw std::invalid_argument{
          "Runtime history requires a non-empty window large enough for its "
          "maximum measured runs."};
    if (m_options.ciCheckInterval == 0u || !std::isfinite(m_options.ciZScore) ||
        m_options.ciZScore <= 0.0 ||
        !std::isfinite(m_options.ciRelativeWidth) ||
        m_options.ciRelativeWidth <= 0.0 ||
        !std::isfinite(m_options.outlierMadScale) ||
        m_options.outlierMadScale <= 0.0)
      throw std::invalid_argument{
          "Runtime history statistical controls must be positive and finite."};
  }

  void rebuildAcceptedSamples() {
    m_acceptedSamples = m_samples;
    if (m_samples.size() < 5u)
      return;

    auto sorted = m_samples;
    std::sort(sorted.begin(), sorted.end());
    auto const median = medianOfSorted(sorted);
    auto deviations = std::vector<double>{};
    deviations.reserve(m_samples.size());
    for (auto const sample : m_samples)
      deviations.push_back(std::abs(sample - median));
    std::sort(deviations.begin(), deviations.end());
    auto const mad = medianOfSorted(deviations);
    auto const robustDeviation = 1.4826 * mad;
    auto const threshold = robustDeviation == 0.0
                               ? std::numeric_limits<double>::epsilon() *
                                     std::max(1.0, std::abs(median))
                               : m_options.outlierMadScale * robustDeviation;

    m_acceptedSamples.clear();
    for (auto const sample : m_samples) {
      if (std::abs(sample - median) <= threshold)
        m_acceptedSamples.push_back(sample);
    }
    // A pathological filter must never make the record unusable.
    if (m_acceptedSamples.empty())
      m_acceptedSamples = m_samples;
  }

  RuntimeHistoryOptions m_options;
  ConfigurationState m_state{ConfigurationState::unmeasured};
  RuntimeCompletion m_completion{RuntimeCompletion::none};
  std::size_t m_warmupRemaining{};
  std::size_t m_currentRunSampleCount{};
  std::vector<double> m_samples;
  std::vector<double> m_acceptedSamples;
};

/**
 * Directional two-sample Mann-Whitney U comparison.
 *
 * For non-tied histories of at most 20 samples in total, the one-sided p
 * values are exact. Larger or tied histories use a continuity- and
 * tie-corrected normal approximation. Lower runtimes are better.
 */
inline auto mannWhitneyUCompare(RuntimeHistory const &lhs,
                                RuntimeHistory const &rhs,
                                std::size_t minimumSamples = 8u,
                                double alpha = 0.05) -> RuntimeComparison {
  auto const lhsValues = lhs.acceptedSamples();
  auto const rhsValues = rhs.acceptedSamples();
  if (minimumSamples == 0u || !std::isfinite(alpha) || alpha <= 0.0 ||
      alpha >= 1.0)
    throw std::invalid_argument{
        "Mann-Whitney U requires positive sample and valid alpha controls."};
  if (lhsValues.size() < minimumSamples || rhsValues.size() < minimumSamples)
    return RuntimeComparison::unavailable;

  auto ranked = std::vector<std::pair<double, int>>{};
  ranked.reserve(lhsValues.size() + rhsValues.size());
  for (auto const value : lhsValues)
    ranked.emplace_back(value, 0);
  for (auto const value : rhsValues)
    ranked.emplace_back(value, 1);
  std::sort(ranked.begin(), ranked.end(),
            [](auto const &left, auto const &right) {
              return left.first < right.first;
            });

  auto lhsRankSum = 0.0;
  auto tieCorrection = 0.0;
  auto hasTies = false;
  for (std::size_t begin = 0u; begin < ranked.size();) {
    auto end = begin + 1u;
    while (end < ranked.size() &&
           ranked.at(end).first == ranked.at(begin).first)
      ++end;
    auto const count = end - begin;
    auto const rank =
        (static_cast<double>(begin + 1u) + static_cast<double>(end)) / 2.0;
    for (auto index = begin; index < end; ++index) {
      if (ranked.at(index).second == 0)
        lhsRankSum += rank;
    }
    tieCorrection += static_cast<double>(count * count * count - count);
    hasTies = hasTies || count > 1u;
    begin = end;
  }

  auto const lhsCount = lhsValues.size();
  auto const rhsCount = rhsValues.size();
  auto const totalCount = lhsCount + rhsCount;
  auto const lhsU =
      lhsRankSum - static_cast<double>(lhsCount * (lhsCount + 1u)) / 2.0;
  auto lowerTail = 1.0;
  auto upperTail = 1.0;

  if (!hasTies && totalCount <= 20u) {
    auto lowerCount = std::size_t{};
    auto upperCount = std::size_t{};
    auto totalPermutations = std::size_t{};
    auto const observedU = static_cast<std::size_t>(std::llround(lhsU));
    auto enumerate = [&](auto const &self, std::size_t nextRank,
                         std::size_t selected, std::size_t rankSum) -> void {
      if (selected == lhsCount) {
        auto const u = rankSum - lhsCount * (lhsCount + 1u) / 2u;
        ++totalPermutations;
        if (u <= observedU)
          ++lowerCount;
        if (u >= observedU)
          ++upperCount;
        return;
      }
      if (totalCount - nextRank < lhsCount - selected)
        return;
      self(self, nextRank + 1u, selected + 1u, rankSum + nextRank + 1u);
      self(self, nextRank + 1u, selected, rankSum);
    };
    enumerate(enumerate, 0u, 0u, 0u);
    lowerTail = static_cast<double>(lowerCount) /
                static_cast<double>(totalPermutations);
    upperTail = static_cast<double>(upperCount) /
                static_cast<double>(totalPermutations);
  } else {
    auto const left = static_cast<double>(lhsCount);
    auto const right = static_cast<double>(rhsCount);
    auto const total = static_cast<double>(totalCount);
    auto const tieFactor =
        1.0 - tieCorrection / (total * total * total - total);
    auto const variance = left * right * (total + 1.0) / 12.0 * tieFactor;
    if (variance <= 0.0)
      return RuntimeComparison::inconclusive;
    auto const mean = left * right / 2.0;
    auto const deviation = std::sqrt(variance);
    auto const normalCdf = [](double value) {
      return 0.5 * std::erfc(-value / std::sqrt(2.0));
    };
    lowerTail = normalCdf((lhsU + 0.5 - mean) / deviation);
    upperTail = 1.0 - normalCdf((lhsU - 0.5 - mean) / deviation);
  }

  if (lowerTail <= alpha)
    return RuntimeComparison::faster;
  if (upperTail <= alpha)
    return RuntimeComparison::slower;
  return RuntimeComparison::inconclusive;
}

} // namespace alpakaTune::detail
