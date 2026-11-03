// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0
#pragma once

#include <alpakaMetrics/alpakaMetrics.hpp>
#include <alpakaTune/alpakaTune.hpp>
#include <alpakaTune/metrics/MetricMeasurement.hpp>

#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

/** Optional, explicit launch measurement and scoring integration. */
namespace alpakaTune::metrics {

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

/** @brief alpakaMetrics plugin: Tuner::enqueue measures through the queue.
 *
 * With this header included, tuner.enqueue(metricsQueue, spec, kernel, ...)
 * launches the selected kernel once, scores its measurement with the tuner's
 * customMetric() objective, and clears the queue's retained measurements. A
 * tuner using the default runtime metric times the launch as usual.
 */
template <typename Queue>
struct alpakaTune::plugin::Enqueue<alpakaMetrics::Queue<Queue>> {
  static constexpr bool enabled = true;

  static void enqueue(auto &tuner, alpakaMetrics::Queue<Queue> const &queue,
                      auto const &spec, auto const &kernelBundle) {
    if constexpr (std::remove_cvref_t<decltype(tuner)>::usesCustomMetric) {
      auto const launch = metrics::enqueue(tuner, queue, spec, kernelBundle);
      metrics::provideMetrics(tuner, launch);
      queue.clearMeasurements();
    } else {
      // Default timing still launches through the metrics queue; the
      // forwarding queue only keeps this specialization from recursing.
      std::optional<alpakaMetrics::Measurement> measurement;
      tuner.enqueue(metrics::detail::CapturingQueue{queue, measurement}, spec,
                    kernelBundle);
    }
  }
};
