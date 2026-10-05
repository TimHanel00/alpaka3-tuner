// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "alpakaTune/tunable/Tunables.hpp"

#include <alpaka/Vec.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace alpakaTune::generate {

namespace detail {

template <alpaka::concepts::VectorOrScalar T>
[[nodiscard]] constexpr auto isPositive(T const &value) -> bool {
  if constexpr (alpaka::concepts::Vector<T>) {
    for (std::size_t dimension = 0u; dimension < T::dim(); ++dimension)
      if (value[dimension] <= 0)
        return false;
    return true;
  } else {
    return value > 0;
  }
}

template <alpaka::concepts::VectorOrScalar T>
[[nodiscard]] constexpr auto isWithin(T const &value, T const &maximum)
    -> bool {
  if constexpr (alpaka::concepts::Vector<T>) {
    for (std::size_t dimension = 0u; dimension < T::dim(); ++dimension)
      if (value[dimension] > maximum[dimension])
        return false;
    return true;
  } else {
    return value <= maximum;
  }
}

template <alpaka::concepts::VectorOrScalar T>
constexpr void add(T &value, T const &step) {
  if constexpr (alpaka::concepts::Vector<T>) {
    for (std::size_t dimension = 0u; dimension < T::dim(); ++dimension)
      value[dimension] += step[dimension];
  } else {
    value += step;
  }
}

template <alpaka::concepts::VectorOrScalar T>
constexpr void multiply(T &value, T const &factor) {
  if constexpr (alpaka::concepts::Vector<T>) {
    for (std::size_t dimension = 0u; dimension < T::dim(); ++dimension)
      value[dimension] *= factor[dimension];
  } else {
    value *= factor;
  }
}

} // namespace detail

template <alpaka::concepts::VectorOrScalar T>
[[nodiscard]] auto linSpace(T first, T last, T step) -> RVals<T> {
  static_assert(!alpaka::isCVector_v<T>,
                "generate::linSpace creates runtime candidates; use CVals or "
                "CTypes for CVec values.");
  if (!detail::isPositive(step))
    throw std::invalid_argument{"generate::linSpace needs a positive step."};
  auto values = std::vector<T>{};
  for (auto value = first; detail::isWithin(value, last);
       detail::add(value, step))
    values.push_back(value);
  return RVals<T>{std::move(values)};
}

template <alpaka::concepts::VectorOrScalar T>
[[nodiscard]] auto logSpace(T first, T last, T factor) -> RVals<T> {
  static_assert(!alpaka::isCVector_v<T>,
                "generate::logSpace creates runtime candidates; use CVals or "
                "CTypes for CVec values.");
  if (!detail::isPositive(first) || !detail::isPositive(factor))
    throw std::invalid_argument{
        "generate::logSpace needs positive values and factors."};
  if constexpr (alpaka::concepts::Vector<T>) {
    for (std::size_t dimension = 0u; dimension < T::dim(); ++dimension)
      if (factor[dimension] <= 1)
        throw std::invalid_argument{
            "generate::logSpace needs factors greater than one."};
  } else if (factor <= 1) {
    throw std::invalid_argument{
        "generate::logSpace needs a factor greater than one."};
  }
  auto values = std::vector<T>{};
  for (auto value = first; detail::isWithin(value, last);
       detail::multiply(value, factor))
    values.push_back(value);
  return RVals<T>{std::move(values)};
}

namespace detail {
template <auto Minimum, auto Maximum, std::size_t... I>
consteval auto compileLinear(std::index_sequence<I...>) {
  static_assert(Minimum <= Maximum);
  return CVals<static_cast<decltype(Minimum)>(
      Minimum + (static_cast<long double>(Maximum) - Minimum) * I /
                    (sizeof...(I) > 1u ? sizeof...(I) - 1u : 1u))...>{};
}
template <auto Minimum, auto Maximum, auto Current, auto... Values>
consteval auto compileLog() {
  static_assert(Minimum > 0 && Minimum <= Maximum);
  if constexpr (Current >= Maximum)
    return CVals<Values..., Maximum>{};
  else if constexpr (Current > Maximum / 2)
    return CVals<Values..., Current, Maximum>{};
  else
    return compileLog<Minimum, Maximum, Current * 2, Values..., Current>();
}
} // namespace detail
} // namespace alpakaTune::generate

namespace alpakaTune::domain {
/** Generate at most eight linearly spaced compiled alternatives from bounds. */
template <auto Minimum, auto Maximum>
[[nodiscard]] consteval auto compileInterval(hint::Linear = {}) {
  static_assert(std::integral<decltype(Minimum)> &&
                std::integral<decltype(Maximum)>);
  static_assert(Minimum <= Maximum);
  constexpr auto count = static_cast<std::size_t>(
      std::min(8.0L, static_cast<long double>(Maximum) - Minimum + 1.0L));
  return generate::detail::compileLinear<Minimum, Maximum>(
      std::make_index_sequence<count>{});
}
/** Generate powers of two relative to the lower bound, including both bounds.
 */
template <auto Minimum, auto Maximum>
[[nodiscard]] consteval auto compileInterval(hint::Logarithmic) {
  return generate::detail::compileLog<Minimum, Maximum, Minimum>();
}
} // namespace alpakaTune::domain

namespace alpakaTune {
/** Concise automatic compiled domain declared using bounds only. */
template <auto Minimum, auto Maximum, typename... Hints>
[[nodiscard]] auto autoCandidates(Hints... hints) {
  if constexpr ((std::same_as<std::remove_cvref_t<Hints>, hint::Logarithmic> ||
                 ... || false))
    return autoCandidates(
        domain::compileInterval<Minimum, Maximum>(hint::logarithmic), hints...);
  else
    return autoCandidates(domain::compileInterval<Minimum, Maximum>(),
                          hints...);
}
} // namespace alpakaTune
