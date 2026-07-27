// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "alpakaTune/core/timing/Timing.hpp"

#include <alpaka/alpaka.hpp>

namespace alpakaTune::detail::timing::internal {

/** Backend customization point for constructing a kernel timer. */
struct MakeKernelTimer {
  template <alpaka::concepts::Api Api,
            alpaka::concepts::DeviceKind DeviceKind>
  struct Op;
};

/** Temporary customization point for a queue with backend timing support. */
struct MakeTuningQueue {
  template <alpaka::concepts::Api Api,
            alpaka::concepts::DeviceKind DeviceKind>
  struct Op;
};

/** Construct the timer associated with an Alpaka API/device-kind pair. */
[[nodiscard]] inline auto
makeKernelTimer(alpaka::onHost::concepts::Device auto &device) {
  return MakeKernelTimer::Op<ALPAKA_TYPEOF(device.getApi()),
                             ALPAKA_TYPEOF(device.getDeviceKind())>{}(device);
}

/** Construct a non-blocking queue suitable for timed tuner launches. */
[[nodiscard]] inline auto
makeTuningQueue(alpaka::onHost::concepts::Device auto &device,
                alpaka::concepts::QueueKind auto kind,
                ::alpakaTune::timing::Enabled timingMode) {
  return MakeTuningQueue::Op<ALPAKA_TYPEOF(device.getApi()),
                             ALPAKA_TYPEOF(device.getDeviceKind())>{}(
      device, kind, timingMode);
}

} // namespace alpakaTune::detail::timing::internal
