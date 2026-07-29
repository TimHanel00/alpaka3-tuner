// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/tag.hpp>

namespace alpakaTune::timing {

using Enabled = alpaka::timing::Enabled;
using Disabled = alpaka::timing::Disabled;

inline constexpr auto enabled = alpaka::timing::enabled;
inline constexpr auto disabled = alpaka::timing::disabled;

template <typename Type>
concept Timing = alpaka::concepts::Timing<Type>;

} // namespace alpakaTune::timing
