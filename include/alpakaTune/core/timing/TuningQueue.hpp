// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpakaTune/core/timing/Timing.hpp>

#include <alpaka/alpaka.hpp>

#include <concepts>

namespace alpakaTune {

/**
 * Construct an Alpaka queue with an explicit timing policy.
 *
 * @note This source-compatible forwarding factory uses Alpaka's public tagged
 * queue interface. New code may call device.makeQueue() directly.
 */
[[nodiscard]] inline auto
makeQueue(alpaka::onHost::concepts::Device auto &device,
          alpaka::concepts::QueueKind auto kind,
          timing::Timing auto timingMode) {
  if constexpr (std::same_as<ALPAKA_TYPEOF(timingMode), timing::Enabled>)
    static_assert(
        std::same_as<ALPAKA_TYPEOF(kind), alpaka::queueKind::NonBlocking>,
        "Timed tuning queues must be non-blocking");
  return device.makeQueue(kind, timingMode);
}

} // namespace alpakaTune
