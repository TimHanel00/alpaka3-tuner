// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <stdexcept>
#include <string>
#include <utility>

namespace alpakaTune {

namespace metric {

/** @brief Compile-time policy selecting built-in kernel-runtime measurement. */
struct Timing {};

/** @brief Compile-time policy selecting application-provided measurements. */
struct Custom {
  /** Runtime label written to tuner diagnostics and persistent histories. */
  std::string name;
};

} // namespace metric

/** @brief Select an application-provided minimization metric for a tuner.
 *
 * Pass the returned token among the identity arguments of makeTuner(). Its
 * type changes the tuner at compile time; its name is runtime metadata and is
 * included in the persistent workload identity.
 */
[[nodiscard]] inline auto customMetric(std::string name) -> metric::Custom {
  if (name.empty())
    throw std::invalid_argument{"A custom tuning metric needs a name."};
  return metric::Custom{std::move(name)};
}

} // namespace alpakaTune
