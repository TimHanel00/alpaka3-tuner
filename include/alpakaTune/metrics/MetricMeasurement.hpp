// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0
#pragma once

#include "alpakaTune/core/TunerConfig.hpp"

#include <alpakaMetrics/alpakaMetrics.hpp>

#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/** Built-in alpakaMetrics measurement selected by TunerConfig::metric. */
namespace alpakaTune::metrics {

[[nodiscard]] constexpr auto statusName(alpakaMetrics::MetricStatus status)
    -> char const * {
  switch (status) {
  case alpakaMetrics::MetricStatus::available:
    return "available";
  case alpakaMetrics::MetricStatus::unsupported:
    return "unsupported";
  case alpakaMetrics::MetricStatus::dependencyDisabled:
    return "dependency_disabled";
  case alpakaMetrics::MetricStatus::permissionDenied:
    return "permission_denied";
  case alpakaMetrics::MetricStatus::conflicting:
    return "conflicting";
  case alpakaMetrics::MetricStatus::unsupportedScope:
    return "unsupported_scope";
  case alpakaMetrics::MetricStatus::collectionFailed:
    return "collection_failed";
  }
  return "unknown";
}

[[nodiscard]] constexpr auto scopeName(alpakaMetrics::MetricScope scope)
    -> char const * {
  switch (scope) {
  case alpakaMetrics::MetricScope::hostRegion:
    return "host_region";
  case alpakaMetrics::MetricScope::callingThread:
    return "calling_thread";
  case alpakaMetrics::MetricScope::queueWorkerThread:
    return "queue_worker_thread";
  case alpakaMetrics::MetricScope::queueInterval:
    return "queue_interval";
  case alpakaMetrics::MetricScope::device:
    return "device";
  case alpakaMetrics::MetricScope::context:
    return "context";
  case alpakaMetrics::MetricScope::providerDefined:
    return "provider_defined";
  }
  return "unknown";
}

/** One available, finite, non-negative value with the expected attribution.
 * An unavailable counter is an error, never zero or a runtime fallback. */
[[nodiscard]] inline auto
metricValue(alpakaMetrics::Result const &result, std::string_view name,
            alpakaMetrics::MetricUnit unit, alpakaMetrics::MetricScope scope)
    -> double {
  auto const &metric = result.getMetric(name);
  if (!metric.isAvailable())
    throw std::runtime_error{std::string{name} + " unavailable: status=" +
                             statusName(metric.status) +
                             "; scope=" + scopeName(metric.descriptor.scope) +
                             "; " + metric.diagnostic};
  if (metric.descriptor.unit != unit || metric.descriptor.scope != scope)
    throw std::runtime_error{std::string{name} +
                             " has incompatible units or attribution scope"};
  auto const value = metric.asDouble();
  if (!std::isfinite(value) || value < 0.0)
    throw std::runtime_error{std::string{name} + " has an invalid value"};
  return value;
}

/** Elapsed queue-interval seconds of a measurement. */
[[nodiscard]] inline auto elapsedSeconds(alpakaMetrics::Result const &result)
    -> double {
  return metricValue(result, "elapsed_time", alpakaMetrics::MetricUnit::seconds,
                     alpakaMetrics::MetricScope::queueInterval);
}

/** Score of a measurement under the configured metric; lower is better. */
[[nodiscard]] inline auto score(alpakaMetrics::Result const &result,
                                TunerConfig const &config) -> double {
  auto counter = [&](Metric metric) {
    return metricValue(result, metricName(metric),
                       alpakaMetrics::MetricUnit::count,
                       alpakaMetrics::MetricScope::queueWorkerThread);
  };
  switch (config.metric) {
  case Metric::runtime:
  case Metric::elapsedTime:
    return elapsedSeconds(result);
  case Metric::instructions:
  case Metric::l2Misses:
    return counter(config.metric);
  case Metric::weighted: {
    auto const &settings = config.weightedMetric;
    return settings.timeWeight * elapsedSeconds(result) /
               settings.timeScale.value() +
           settings.counterWeight * counter(settings.counter) /
               settings.counterScale.value();
  }
  }
  throw std::invalid_argument{"Unknown tuning metric."};
}

/** alpakaMetrics requests needed to score the configured metric. */
[[nodiscard]] inline auto requests(TunerConfig const &config)
    -> std::vector<alpakaMetrics::MetricRequest> {
  auto result = std::vector<alpakaMetrics::MetricRequest>{
      alpakaMetrics::metric::elapsedTime};
  auto const counter = config.metric == Metric::weighted
                           ? config.weightedMetric.counter
                           : config.metric;
  if (counter == Metric::instructions)
    result.emplace_back(alpakaMetrics::metric::instructions);
  else if (counter == Metric::l2Misses)
    result.emplace_back(alpakaMetrics::metric::l2Misses);
  return result;
}

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

/** @brief The alpakaMetrics queue a tuner measures through.
 *
 * It wraps the application's queue once and is rebuilt only when the tuner is
 * enqueued on a different queue, so counter setup is not repeated per launch.
 */
class MeasurementQueue {
public:
  /** Elapsed seconds and score of one launch of the selected kernel. */
  struct Sample {
    double seconds;
    double score;
  };

  template <typename Queue>
  [[nodiscard]] auto measure(TunerConfig const &config, Queue const &queue,
                             std::string const &label, auto &&launch)
      -> Sample {
    if constexpr (!requires { alpakaMetrics::makeQueue(queue); }) {
      throw std::invalid_argument{
          "TunerConfig::metric needs an alpaka queue; pass the underlying "
          "queue instead of an alpakaMetrics queue."};
    } else {
      using MetricsQueue = decltype(alpakaMetrics::makeQueue(queue));
      auto *metricsQueue = static_cast<MetricsQueue *>(m_queue.get());
      if (!metricsQueue || !(metricsQueue->getUnderlyingQueue() == queue)) {
        auto created = std::make_shared<MetricsQueue>(alpakaMetrics::makeQueue(
            queue, alpakaMetrics::Config{.metrics = requests(config),
                                         .label = label}));
        metricsQueue = created.get();
        m_queue = std::move(created);
      }
      std::optional<alpakaMetrics::Measurement> measurement;
      launch(CapturingQueue{*metricsQueue, measurement});
      if (!measurement)
        throw std::logic_error{"Tuner launch did not produce a measurement"};
      auto const result = measurement->getResults();
      metricsQueue->clearMeasurements();
      return {.seconds = elapsedSeconds(result),
              .score = score(result, config)};
    }
  }

private:
  std::shared_ptr<void> m_queue;
};
} // namespace detail
} // namespace alpakaTune::metrics
