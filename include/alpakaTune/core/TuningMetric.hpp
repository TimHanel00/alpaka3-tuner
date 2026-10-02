// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <concepts>
#include <stdexcept>
#include <string>
#include <type_traits>
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

/** @brief Application scoring function, evaluated only by provideMetrics(). */
template <typename Function> struct FunctionObjective {
  std::string name;
  Function function;
};

} // namespace metric

namespace detail {
template <typename T> inline constexpr bool isCustomMetricToken = false;
template <> inline constexpr bool isCustomMetricToken<metric::Custom> = true;
template <typename Function>
inline constexpr bool isCustomMetricToken<metric::FunctionObjective<Function>> =
    true;
// Specialize before inspecting the callable, so timing/scalar policies never
// instantiate an expression referring to a nonexistent function member.
template <typename Policy, typename... Values>
inline constexpr bool acceptsMetricInputs = false;
template <typename Function, typename... Values>
inline constexpr bool
    acceptsMetricInputs<metric::FunctionObjective<Function>, Values...> =
        std::is_invocable_r_v<double, Function &, Values...>;
} // namespace detail

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

/** @brief Select a user scoring function with explicit metric submission.
 * Captures own weights/scales. Include their configuration in workload
 * identity.
 */
template <typename Function>
  requires std::copy_constructible<std::decay_t<Function>>
[[nodiscard]] auto customMetric(std::string name, Function &&function) {
  auto token = customMetric(std::move(name));
  return metric::FunctionObjective<std::decay_t<Function>>{
      std::move(token.name), std::forward<Function>(function)};
}

} // namespace alpakaTune
