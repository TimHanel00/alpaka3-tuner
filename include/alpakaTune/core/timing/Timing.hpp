// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <type_traits>

namespace alpakaTune::timing {

namespace detail {
struct TimingBase {};
} // namespace detail

/** Request backend timing support from a queue or event. */
struct Enabled : detail::TimingBase {};

/** Do not request backend timing support. */
struct Disabled : detail::TimingBase {};

inline constexpr auto enabled = Enabled{};
inline constexpr auto disabled = Disabled{};

template <typename Type>
concept Timing = std::is_base_of_v<detail::TimingBase, Type>;

} // namespace alpakaTune::timing
