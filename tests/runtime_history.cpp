// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#include <alpakaTune/store/RuntimeHistory.hpp>

#include <array>
#include <cmath>
#include <cstdlib>

auto main() -> int {
  using alpakaTune::ConfigurationState;
  using alpakaTune::RuntimeComparison;
  using alpakaTune::detail::RuntimeHistory;
  using alpakaTune::detail::RuntimeHistoryOptions;

  auto const options =
      RuntimeHistoryOptions{0u, 5u, 20u, 5u, 2.576, 0.05, 3.5, 20u};
  auto history = RuntimeHistory{options};
  history.beginActivation();
  for (auto const runtime : std::array{10.0, 10.05, 9.95, 10.0, 1'000.0, 10.02,
                                       9.98, 10.01, 9.99, 10.0})
    static_cast<void>(history.record(runtime));
  auto const statistics = history.statistics();
  if (!history.isFinished() || history.state() != ConfigurationState::retired ||
      statistics.sampleCount != 10u || statistics.acceptedSampleCount != 9u ||
      !statistics.confidenceReached ||
      std::abs(statistics.estimate() - 10.0) > 0.03 ||
      statistics.rawMean < 100.0)
    return EXIT_FAILURE;

  auto fast = RuntimeHistory{
      RuntimeHistoryOptions{0u, 20u, 20u, 10u, 2.576, 0.05, 3.5, 20u}};
  auto slow = RuntimeHistory{
      RuntimeHistoryOptions{0u, 20u, 20u, 10u, 2.576, 0.05, 3.5, 20u}};
  fast.beginActivation();
  slow.beginActivation();
  for (auto index = 0u; index < 8u; ++index) {
    static_cast<void>(fast.record(9.8 + static_cast<double>(index) * 0.05));
    static_cast<void>(slow.record(19.8 + static_cast<double>(index) * 0.05));
  }
  if (alpakaTune::detail::mannWhitneyUCompare(fast, slow) !=
          RuntimeComparison::faster ||
      alpakaTune::detail::mannWhitneyUCompare(slow, fast) !=
          RuntimeComparison::slower ||
      alpakaTune::detail::mannWhitneyUCompare(fast, slow, 9u) !=
          RuntimeComparison::unavailable)
    return EXIT_FAILURE;
  for (auto index = 8u; index < 11u; ++index) {
    static_cast<void>(fast.record(9.8 + static_cast<double>(index) * 0.05));
    static_cast<void>(slow.record(19.8 + static_cast<double>(index) * 0.05));
  }
  if (alpakaTune::detail::mannWhitneyUCompare(fast, slow) !=
          RuntimeComparison::faster ||
      alpakaTune::detail::mannWhitneyUCompare(slow, fast) !=
          RuntimeComparison::slower)
    return EXIT_FAILURE;

  auto warmed =
      RuntimeHistory{RuntimeHistoryOptions{2u, 1u, 2u, 1u, 2.576, 0.05, 3.5}};
  warmed.beginActivation();
  static_cast<void>(warmed.record(1.0));
  static_cast<void>(warmed.record(1.0));
  if (warmed.state() != ConfigurationState::measuring || !warmed.empty())
    return EXIT_FAILURE;
  static_cast<void>(warmed.record(1.0));
  if (warmed.statistics().sampleCount != 1u)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
