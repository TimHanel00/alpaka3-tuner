/* Copyright 2025 Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once
#include "alpakaTune/concepts.hpp"

namespace aTune::internal {
template <concepts::Serializable T>
std::string toStringGeneric(T const &value) {
  if constexpr (concepts::serialize::HasTraitSerializer<T>) {
    return trait::Serialize<T>{}(value);
  } else if constexpr (std::is_convertible_v<T, std::string>) {
    return std::string(value);
  } else if constexpr (std::is_arithmetic_v<T> ||
                       concepts::serialize::HasStdToString<T>) {
    return std::to_string(value);
  } else if constexpr (concepts::serialize::HasToStringMethod<T>) {
    return value.toString();
  } else if constexpr (concepts::serialize::HasStreamOperator<T>) {
    std::ostringstream oss;
    oss << value;
    return oss.str();
  } else {
    static_assert(!std::is_same_v<T, T>, "Could not convert type to string! ");
  }
  return "";
}
} // namespace aTune::internal
