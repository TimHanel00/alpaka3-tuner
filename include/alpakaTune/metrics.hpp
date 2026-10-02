// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0
#pragma once

#include <alpakaMetrics/alpakaMetrics.hpp>
#include <alpakaTune/alpakaTune.hpp>

#include <memory>
#include <optional>
#include <stdexcept>

/** Optional, explicit launch measurement and scoring integration. */
namespace alpakaTune::metrics {
namespace detail {
/** Captures the selected kernel's returned handle, without inspecting history.
 */
template <typename Queue> class CapturingQueue {
public:
  CapturingQueue(Queue const &queue,
                 std::optional<alpakaMetrics::Measurement> &measurement)
      : m_queue(queue), m_measurement(measurement) {}
  auto getDevice() const { return m_queue.getDevice(); }
  auto getTiming() const { return m_queue.getTiming(); }
  auto getQueueKind() const { return m_queue.getQueueKind(); }

  void enqueue(alpaka::onHost::concepts::ThreadOrFrameSpec auto const &spec,
               alpaka::concepts::KernelBundle auto const &bundle) const {
    m_measurement.emplace(m_queue.enqueue(spec, bundle));
  }
  void enqueue(auto const &event) const { m_queue.enqueue(event); }

private:
  Queue const &m_queue;
  std::optional<alpakaMetrics::Measurement> &m_measurement;
};
} // namespace detail

/** Stable handle tied to the tuner execution that produced it. */
template <typename Tuner> class MeasuredLaunch {
public:
  MeasuredLaunch(Tuner const &tuner, std::size_t executionIndex,
                 alpakaMetrics::Measurement measurement)
      : m_tuner(std::addressof(tuner)), m_executionIndex(executionIndex),
        m_measurement(std::move(measurement)) {}

  auto getResults() const { return m_measurement.getResults(); }
  auto getId() const { return m_measurement.getId(); }
  void validate(Tuner const &tuner) const {
    if (m_tuner != std::addressof(tuner) || tuner.history().empty() ||
        tuner.lastConfig().executionIndex != m_executionIndex)
      throw std::logic_error{
          "Measurement does not belong to this tuner's latest launch"};
  }

private:
  Tuner const *m_tuner;
  std::size_t m_executionIndex;
  alpakaMetrics::Measurement m_measurement;
};

/** Submit exactly one selected kernel through the configured metrics queue. */
template <typename Tuner, typename Queue>
  requires(Tuner::usesCustomMetric)
[[nodiscard]] auto
enqueue(Tuner &tuner, alpakaMetrics::Queue<Queue> const &queue,
        alpaka::onHost::concepts::ThreadOrFrameSpec auto const &spec,
        alpaka::concepts::KernelBundle auto const &bundle) {
  std::optional<alpakaMetrics::Measurement> measurement;
  auto capturingQueue = detail::CapturingQueue{queue, measurement};
  tuner.enqueue(capturingQueue, spec, bundle);
  if (!measurement)
    throw std::logic_error{"Tuner launch did not produce a measurement"};
  return MeasuredLaunch<Tuner>{tuner, tuner.lastConfig().executionIndex,
                               std::move(*measurement)};
}

/** Explicitly complete and score the latest measurement; no automatic fallback.
 */
template <typename Tuner>
  requires requires(Tuner &tuner, alpakaMetrics::Result const &result) {
    tuner.provideMetrics(result);
  }
void provideMetrics(Tuner &tuner, MeasuredLaunch<Tuner> const &launch) {
  launch.validate(tuner);
  tuner.provideMetrics(launch.getResults());
}
} // namespace alpakaTune::metrics
