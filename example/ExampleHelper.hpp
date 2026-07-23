// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <cstddef>

/** @file
 * @brief Internal conveniences shared by the repository's example programs.
 *
 * This header is not part of the alpakaTune library interface and is not
 * installed. It only removes repetition between examples. Applications using
 * alpakaTune should define their own launch counts and loop conditions.
 */

namespace alpakaTune::example {

/** @brief Convenience launch minimum shared by the example programs.
 *
 * This is not part of the alpakaTune interface and applications do not need to
 * use it. Each application remains responsible for its own launch count. The
 * examples choose 50,000 independently of TunerConfig::horizon so
 * that the default 40,000-execution horizon leaves a 10,000-launch diagnostic
 * tail.
 */
inline constexpr std::size_t minimumTuningExecutions = 50'000u;

/** @brief Apply the example's minimum-launch plus tuner-policy condition.
 *
 * @param completedExecutions Launches already requested by the application.
 * @param minimumExecutions Application-owned minimum number of launches.
 * @param tuners Tuning contexts whose optional policy goals the example chose
 * to observe.
 *
 * @return true while the application minimum or any policy goal is pending.
 *
 * This predicate does not control a tuner and is not required by alpakaTune.
 * It spells the following example-owned condition without repeating it:
 *
 * @code
 * completedExecutions < minimumExecutions ||
 *     !firstTuner.completed() || !secondTuner.completed()
 * @endcode
 */
template <typename... Tuners>
[[nodiscard]] auto applicationRunsRemain(std::size_t completedExecutions,
                                         std::size_t minimumExecutions,
                                         Tuners const &...tuners) noexcept
    -> bool {
  return completedExecutions < minimumExecutions ||
         (... || !tuners.completed());
}

/** @brief Use the examples' shared 50,000-launch convenience minimum. */
template <typename... Tuners>
[[nodiscard]] auto
applicationRunsRemain(std::size_t completedExecutions,
                      Tuners const &...tuners) noexcept -> bool {
  return applicationRunsRemain(completedExecutions, minimumTuningExecutions,
                               tuners...);
}

} // namespace alpakaTune::example
