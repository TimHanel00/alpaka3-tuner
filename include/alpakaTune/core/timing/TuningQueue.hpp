// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpakaTune/core/timing/KernelTimer.hpp>

#include <concepts>

namespace alpakaTune {

/**
 * Construct an Alpaka queue with an explicit timing policy.
 *
 * @note `timing::enabled` currently requires `queueKind::nonBlocking` and
 * supplies a profiling-enabled compatibility queue for SYCL GPUs. This
 * temporary factory will be replaced by Alpaka's public tagged interface.
 */
[[nodiscard]] inline auto
makeQueue(alpaka::onHost::concepts::Device auto &device,
          alpaka::concepts::QueueKind auto kind,
          timing::Timing auto timingMode) {
  if constexpr (std::same_as<ALPAKA_TYPEOF(timingMode), timing::Disabled>) {
    return device.makeQueue(kind);
  } else {
    static_assert(std::same_as<ALPAKA_TYPEOF(kind),
                               alpaka::queueKind::NonBlocking>,
                  "Timed tuning queues must be non-blocking");
    return detail::timing::internal::makeTuningQueue(device, kind,
                                                      timingMode);
  }
}

} // namespace alpakaTune
