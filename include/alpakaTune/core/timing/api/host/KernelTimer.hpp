// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "alpakaTune/core/TunerInfo.hpp"
#include "alpakaTune/core/timing/internal/interface.hpp"

#include <alpaka/alpaka.hpp>

#include <chrono>
#include <utility>

namespace alpakaTune::detail::timing::host {

/** Synchronized host-clock fallback for APIs without device timestamps. */
template <alpaka::onHost::concepts::Device Device> class KernelTimer {
public:
  explicit KernelTimer(Device &device)
      : m_startEvent(device.makeEvent()), m_endEvent(device.makeEvent()) {}

  [[nodiscard]] static constexpr auto measurementSource() noexcept
      -> RuntimeMeasurementSource {
    return RuntimeMeasurementSource::hostClock;
  }

  [[nodiscard]] auto measure(auto const &queue, auto &&launch) -> double {
    queue.enqueue(m_startEvent);
    alpaka::onHost::wait(m_startEvent);
    auto const start = std::chrono::steady_clock::now();
    ALPAKA_FORWARD(launch)(queue);
    queue.enqueue(m_endEvent);
    alpaka::onHost::wait(m_endEvent);
    return std::chrono::duration<double>{std::chrono::steady_clock::now() -
                                         start}
        .count();
  }

private:
  ALPAKA_TYPEOF(std::declval<Device &>().makeEvent()) m_startEvent;
  ALPAKA_TYPEOF(std::declval<Device &>().makeEvent()) m_endEvent;
};

} // namespace alpakaTune::detail::timing::host

namespace alpakaTune::detail::timing::internal {

template <alpaka::concepts::DeviceKind DeviceKind>
struct MakeKernelTimer::Op<alpaka::api::Host, DeviceKind> {
  [[nodiscard]] auto operator()(alpaka::onHost::concepts::Device auto &device)
      const {
    return host::KernelTimer<ALPAKA_TYPEOF(device)>{device};
  }
};

template <alpaka::concepts::DeviceKind DeviceKind>
struct MakeTuningQueue::Op<alpaka::api::Host, DeviceKind> {
  [[nodiscard]] auto operator()(alpaka::onHost::concepts::Device auto &device,
                                alpaka::concepts::QueueKind auto kind,
                                ::alpakaTune::timing::Enabled) const {
    return device.makeQueue(kind);
  }
};

} // namespace alpakaTune::detail::timing::internal
