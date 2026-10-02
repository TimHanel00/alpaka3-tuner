// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0
#pragma once

#include <alpakaTune/alpakaTune.hpp>
#ifdef ALPAKA_TUNE_EXAMPLE_METRICS
#include <alpakaTune/metrics.hpp>
#endif

#include <cmath>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace alpakaTune::example {
/** Example-owned settings; this header is not an installed library API. */
struct ObjectiveOptions {
  std::string objective{"runtime"};
  std::string counter{"instructions"};
  double timeWeight{0.5};
  double counterWeight{0.5};
  std::optional<double> timeScale;
  std::optional<double> counterScale;

  bool needsCounter() const {
    return objective == "instructions" || objective == "l2-misses" ||
           objective == "weighted";
  }
  std::string counterName() const {
    return (objective == "weighted" ? counter : objective) == "l2-misses"
               ? "l2_misses"
               : "instructions";
  }
  std::string identity() const {
    std::ostringstream text;
    text << std::setprecision(17) << "queue_objective_v1;" << objective;
    if (needsCounter())
      text << ";counter=" << counterName();
    if (objective == "weighted")
      text << ";weights=" << timeWeight << ',' << counterWeight
           << ";scales=" << timeScale.value() << ',' << counterScale.value();
    return text.str();
  }
  void validate() const {
    if (objective != "runtime" && objective != "elapsed-time" &&
        objective != "instructions" && objective != "l2-misses" &&
        objective != "weighted")
      throw std::invalid_argument{"Unknown --objective"};
    if (counter != "instructions" && counter != "l2-misses")
      throw std::invalid_argument{
          "--counter must be instructions or l2-misses"};
    if (!std::isfinite(timeWeight) || timeWeight < 0.0 ||
        !std::isfinite(counterWeight) || counterWeight < 0.0 ||
        (timeWeight == 0.0 && counterWeight == 0.0))
      throw std::invalid_argument{
          "Weights must be finite, non-negative, and not both zero"};
    if (objective == "weighted" && (!timeScale || !counterScale))
      throw std::invalid_argument{
          "Weighted scoring requires --time-scale-seconds and --counter-scale"};
    for (auto scale : {timeScale, counterScale})
      if (scale && (!std::isfinite(*scale) || *scale <= 0.0))
        throw std::invalid_argument{
            "Normalization scales must be positive and finite"};
#ifndef ALPAKA_TUNE_EXAMPLE_METRICS
    if (objective != "runtime")
      throw std::invalid_argument{
          "This objective requires -DalpakaTune_DEP_METRICS=ON"};
#endif
  }
};
inline auto objectiveOptions() -> ObjectiveOptions & {
  static ObjectiveOptions options;
  return options;
}

inline bool consumeObjectiveOptions(int &argc, char **argv,
                                    std::string defaultCounter) {
  auto &options = objectiveOptions();
  options = ObjectiveOptions{};
  options.counter = std::move(defaultCounter);
  std::size_t destination = 1u;
  try {
    for (int i = 1; i < argc; ++i) {
      auto argument = std::string_view{argv[i]};
      auto separator = argument.find('=');
      auto name = argument.substr(0u, separator);
      if (name != "--objective" && name != "--counter" &&
          name != "--time-weight" && name != "--counter-weight" &&
          name != "--time-scale-seconds" && name != "--counter-scale") {
        argv[destination++] = argv[i];
        continue;
      }
      auto value = std::string{};
      if (separator != std::string_view::npos)
        value = argument.substr(separator + 1u);
      else {
        if (++i == argc)
          throw std::invalid_argument{"Missing value for " + std::string{name}};
        value = argv[i];
      }
      if (name == "--objective")
        options.objective = value;
      else if (name == "--counter")
        options.counter = value;
      else {
        std::size_t used = 0u;
        auto number = std::stod(value, &used);
        if (used != value.size())
          throw std::invalid_argument{"Invalid numeric option: " + value};
        if (name == "--time-weight")
          options.timeWeight = number;
        else if (name == "--counter-weight")
          options.counterWeight = number;
        else if (name == "--time-scale-seconds")
          options.timeScale = number;
        else
          options.counterScale = number;
      }
    }
    options.validate();
    argc = static_cast<int>(destination);
    argv[destination] = nullptr;
    return true;
  } catch (std::exception const &error) {
    std::cerr << "Objective options: " << error.what() << '\n';
    return false;
  }
}
inline void printObjectiveHelp() {
  std::cerr
      << "  --objective runtime|elapsed-time|instructions|l2-misses|weighted\n"
      << "  --counter instructions|l2-misses (weighted objective)\n"
      << "  --time-weight 0.5 --counter-weight 0.5\n"
      << "  --time-scale-seconds VALUE --counter-scale VALUE (required for "
         "weighted)\n";
}

#ifdef ALPAKA_TUNE_EXAMPLE_METRICS
inline auto statusName(alpakaMetrics::MetricStatus status) -> char const * {
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
inline auto scopeName(alpakaMetrics::MetricScope scope) -> char const * {
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
inline auto metricValue(alpakaMetrics::Result const &result,
                        std::string_view name, alpakaMetrics::MetricUnit unit,
                        alpakaMetrics::MetricScope scope) -> double {
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
inline auto scoreMetrics(alpakaMetrics::Result const &result,
                         ObjectiveOptions const &options) -> double {
  if (options.objective == "elapsed-time")
    return metricValue(result, "elapsed_time",
                       alpakaMetrics::MetricUnit::seconds,
                       alpakaMetrics::MetricScope::queueInterval);
  auto const count = metricValue(result, options.counterName(),
                                 alpakaMetrics::MetricUnit::count,
                                 alpakaMetrics::MetricScope::queueWorkerThread);
  if (options.objective != "weighted")
    return count;
  auto const seconds =
      metricValue(result, "elapsed_time", alpakaMetrics::MetricUnit::seconds,
                  alpakaMetrics::MetricScope::queueInterval);
  return options.timeWeight * seconds / options.timeScale.value() +
         options.counterWeight * count / options.counterScale.value();
}
#endif

/** Invoke the existing example loop with either the base or metrics-backed
 * tuner. */
template <typename Tunables, typename Identity, typename Run>
  requires std::constructible_from<std::string, Identity const &>
int withObjective(Tunables const &tunables,
                  alpaka::onHost::concepts::Device auto device,
                  alpaka::concepts::Executor auto executor,
                  Identity const &identity, Run &&run) {
  try {
    auto const options = objectiveOptions();
    if (options.objective == "runtime") {
      auto queue = device.makeQueue(alpaka::queueKind::nonBlocking,
                                    alpaka::timing::enabled);
      auto tuner = alpakaTune::makeTuner(tunables, device, executor, identity);
      return std::forward<Run>(run)(tuner, queue);
    }
#ifdef ALPAKA_TUNE_EXAMPLE_METRICS
    if (options.needsCounter() &&
        (!std::same_as<decltype(device.getApi()), alpaka::api::Host> ||
         !std::same_as<decltype(executor), alpaka::exec::CpuSerial>))
      throw std::invalid_argument{
          "Counter objectives require --backend host:cpu --executor CpuSerial"};
    alpakaMetrics::Config config{
        .metrics = {alpakaMetrics::metric::elapsedTime},
        .label = std::string{identity}};
    if (options.needsCounter()) {
      if (options.counterName() == "instructions")
        config.metrics.emplace_back(alpakaMetrics::metric::instructions);
      else
        config.metrics.emplace_back(alpakaMetrics::metric::l2Misses);
    }
    auto queue = alpakaMetrics::makeQueue(
        device.makeQueue(alpaka::queueKind::nonBlocking,
                         alpaka::timing::enabled),
        config);
    std::optional<alpakaMetrics::Result> latest;
    auto objective = alpakaTune::customMetric(
        "queue_" + options.objective + "_v1",
        [options, &latest](alpakaMetrics::Result const &result) {
          auto score = scoreMetrics(result, options);
          latest = result;
          return score;
        });
    auto tuner = alpakaTune::makeTuner(tunables, device, executor, identity,
                                       objective, options.identity());
    std::cout << "Objective: " << options.identity() << '\n';
    auto status = std::forward<Run>(run)(tuner, queue);
    if (latest) {
      for (auto const &metric : latest->metrics) {
        if (!metric.isAvailable())
          continue;
        std::cout << metric.descriptor.name << '=' << metric.asDouble()
                  << " unit="
                  << (metric.descriptor.unit ==
                              alpakaMetrics::MetricUnit::seconds
                          ? "seconds"
                          : "count")
                  << " scope=" << scopeName(metric.descriptor.scope)
                  << " provider=" << metric.descriptor.provider << '\n';
      }
      std::cout << "Latest score=" << tuner.lastConfig().metricValue.value()
                << '\n';
    }
    return status;
#else
    throw std::invalid_argument{"Metrics integration is disabled"};
#endif
  } catch (std::exception const &error) {
    std::cerr << "Tuning objective failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}

inline auto underlyingQueue(auto const &queue) -> decltype(auto) {
#ifdef ALPAKA_TUNE_EXAMPLE_METRICS
  if constexpr (requires { queue.getUnderlyingQueue(); })
    return queue.getUnderlyingQueue();
  else
#endif
    return queue;
}
inline void
enqueueObjective(auto &tuner, auto const &queue,
                 alpaka::onHost::concepts::ThreadOrFrameSpec auto const &spec,
                 alpaka::concepts::KernelBundle auto const &bundle) {
#ifdef ALPAKA_TUNE_EXAMPLE_METRICS
  if constexpr (std::remove_cvref_t<decltype(tuner)>::usesCustomMetric) {
    auto launch = alpakaTune::metrics::enqueue(tuner, queue, spec, bundle);
    alpakaTune::metrics::provideMetrics(tuner, launch);
    queue.clearMeasurements();
  } else
#endif
  {
    tuner.enqueue(queue, spec, bundle);
  }
}
} // namespace alpakaTune::example
