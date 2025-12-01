/* Copyright 2025 Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once
#include "tune/utils/serialize.hpp"

#include <string>
#include <vector>

namespace tune::internal {

inline void processArgs(std::vector<std::string> &) {}

// Recursive case: contains arithmetric Argument
template <typename First, typename... Rest>
void processArgs(std::vector<std::string> &specifierStrings, First first,
                 Rest... rest) {
  if constexpr (concepts::Serializable<First>) {
    specifierStrings.push_back(
        toStringGeneric(first));            // Convert numbers to strings
    processArgs(specifierStrings, rest...); // Process remaining args
  } else {
    static_assert(!std::is_same_v<First, First>,
                  "All session specifier must not be serializable, see concept "
                  "tune::Serializable! ");
  }
}

// Recursive case: contains string Argument
template <typename... Args>
void processArgs(std::vector<std::string> &specifierStrings,
                 std::string const &str, Args... rest) {
  specifierStrings.push_back(str);
  processArgs(specifierStrings, rest...);
}
} // namespace tune::internal
