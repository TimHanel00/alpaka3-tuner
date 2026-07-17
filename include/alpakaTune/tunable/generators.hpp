// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "alpakaTune/tunable/Tunables.hpp"

#include <alpaka/Vec.hpp>

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

} // namespace alpakaTune::generate
