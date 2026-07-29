// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "alpakaTune/core/TunerInfo.hpp"

#include <alpaka/alpaka.hpp>

#include <concepts>
#include <utility>

namespace alpakaTune::detail::timing {

/** Measure one kernel with Alpaka timing events on a tagged queue. */
template <alpaka::onHost::concepts::Device Device> class KernelTimer {
public:
  explicit KernelTimer(Device &device)
      : m_start(device.makeEvent(alpaka::timing::enabled)),
        m_end(device.makeEvent(alpaka::timing::enabled)) {}

  [[nodiscard]] static constexpr auto measurementSource() noexcept
      -> RuntimeMeasurementSource {
    if constexpr (std::same_as<ALPAKA_TYPEOF(std::declval<Device>().getApi()),
                               alpaka::api::Host>)
      return RuntimeMeasurementSource::hostClock;
    else
      return RuntimeMeasurementSource::deviceEvent;
  }

  [[nodiscard]] auto measure(auto const &queue, auto &&launch) -> double {
    static_assert(std::same_as<ALPAKA_TYPEOF(queue.getQueueKind()),
                               alpaka::queueKind::NonBlocking>,
                  "Tuned kernel timing requires a non-blocking queue.");
    static_assert(
        std::same_as<ALPAKA_TYPEOF(queue.getTiming()), alpaka::timing::Enabled>,
        "Tuned kernel timing requires a timing-enabled queue.");
    queue.enqueue(m_start);
    ALPAKA_FORWARD(launch)(queue);
    queue.enqueue(m_end);
    return alpaka::onHost::getElapsedTime(m_start, m_end).count();
  }

private:
  ALPAKA_TYPEOF(std::declval<Device &>().makeEvent(alpaka::timing::enabled))
  m_start;
  ALPAKA_TYPEOF(std::declval<Device &>().makeEvent(alpaka::timing::enabled))
  m_end;
};

} // namespace alpakaTune::detail::timing
