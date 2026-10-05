// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0
#pragma once

#include "PiCalculator.hpp"
#include <alpakaTune/alpakaTune.hpp>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numbers>
#include <sstream>
#include <string_view>
#include <vector>

namespace adaptivePi {
inline constexpr auto maximumSplits = ALPAKA_TUNE_TUNABLE("maximumSplits");
inline constexpr auto pointsPerTile = ALPAKA_TUNE_TUNABLE("pointsPerTile");
inline auto tuningConfig() -> alpakaTune::TunerConfig {
  auto config = alpakaTune::TunerConfig{};
  config.exploration = alpakaTune::ExplorationPolicy::online;
  config.selection = alpakaTune::SelectionPolicy::fixed;
  config.replayFastPath = true;
  config.strategy = alpakaTune::StrategyKind::exhaustive;
  config.queue.reset();
  config.runsPerCandidate = 1u;
  config.minimumRunsPerCandidate = 1u;
  config.maximumExecutions = 1024u;
  config.maximumRetiredConfigurations.reset();
  config.mannWhitneyEarlyStop = false;
  config.history = {.read = false, .write = false};
  config.completeHistory = {.read = false, .write = false};
  return config;
}

struct Optimization {
  bool feasible{};
  Calculation winner{};
  std::size_t candidateCount{}, rejectedCount{}, replayCount{};
  double bestError{std::numeric_limits<double>::infinity()};
  double bestMetric{std::numeric_limits<double>::infinity()};
  std::array<std::uint32_t, 2u> parameters{};
  struct Trial {
    Calculation calculation;
    double metric;
    bool valid;
    bool replay;
  };
  std::vector<Trial> trials;
};

/** Select which quantity is minimized and which is a post-run constraint. */
enum class Objective { accuracy, runtime };
struct AccuracyObjective {
  static constexpr bool custom = true;
};
struct RuntimeObjective {
  static constexpr bool custom = false;
};

// One calculation engine and one tuning loop serve both objectives. The policy
// only selects the queue instrumentation, tuner metric, and validity decision.
struct IgnoreInspection {
  void operator()(auto &, Calculation &) const {}
};

template <typename Policy, typename Inspector>
auto optimizeImpl(alpaka::onHost::concepts::Device auto device,
                  alpaka::concepts::Executor auto executor,
                  double budgetSeconds, double maximumError, Policy,
                  Inspector const &inspect, std::string_view workload)
    -> Optimization {
  require(std::isfinite(budgetSeconds) && budgetSeconds >= 0.0,
          "Invalid runtime budget");
  require(std::isfinite(maximumError) && maximumError >= 0.0,
          "Invalid accuracy limit");
  auto calculator = Calculator{device, executor};
  auto selector = alpaka::onHost::makeDeviceSelector(
      alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu});
  auto host = selector.makeDevice(0u);
  auto queue = [&] {
    if constexpr (Policy::custom)
      return alpakaTune::makeQueue(host, alpaka::queueKind::nonBlocking,
                                   alpakaTune::timing::disabled);
    else
      return alpakaTune::makeQueue(host, alpaka::queueKind::nonBlocking,
                                   alpakaTune::timing::enabled);
  }();
  // A single host invocation orchestrates the entire calculation. Its inner
  // queues may run on a GPU; calculate() waits for the final reduction.
  // Built-in timing therefore measures the complete workflow, including
  // transfers/refinement.
  auto const frame =
      alpaka::onHost::FrameSpec{Index{1u}, Index{1u}, alpaka::exec::cpuSerial};
  // 4 split depths * 80 sample counts = 320 configurations. Device launch
  // geometry stays fixed; these host-kernel arguments control the algorithm.
  std::vector<std::uint32_t> sampleCounts;
  for (std::uint32_t count = 32u; count <= maximumSamplesPerTile; count += 32u)
    sampleCounts.push_back(count);
  auto const tunables = alpakaTune::TunableBundle{
      maximumSplits(alpakaTune::RVals<std::uint32_t>::list(2u, 4u, 6u, 8u)),
      pointsPerTile(alpakaTune::RVals<std::uint32_t>{std::move(sampleCounts)})};
  // Post-validation decisions belong to this workload's history identity.
  auto identity = std::ostringstream{};
  identity << std::setprecision(std::numeric_limits<double>::max_digits10)
           << workload << ":runtime-budget=" << budgetSeconds
           << ":accuracy-limit=" << maximumError;
  auto tuner = [&] {
    if constexpr (Policy::custom)
      return alpakaTune::makeTuner(
          tuningConfig(), tunables, host,
          alpakaTune::customMetric("absolute_pi_error"), device, executor,
          identity.str());
    else
      return alpakaTune::makeTuner(tuningConfig(), tunables, host, device,
                                   executor, budgetSeconds, maximumError,
                                   workload);
  }();
  auto const bundle = alpaka::KernelBundle{RunCalculation{}, &calculator,
                                           maximumSplits, pointsPerTile};
  auto outcome = Optimization{};
  outcome.candidateCount = tuner.info().candidateCount;
  // Bound the search, including replacement winners rejected during replay.
  constexpr std::size_t winnerReplays = 4u;
  auto const launchLimit = 2u * outcome.candidateCount + winnerReplays;
  for (std::size_t launch = 0u; launch < launchLimit; ++launch) {
    bool const replay = tuner.completed();
    try {
      tuner.enqueue(queue, frame, bundle);
    } catch (std::exception const &) {
      if (tuner.completed() &&
          tuner.completionReason() ==
              alpakaTune::TunerCompletionReason::noValidConfiguration)
        break;
      throw;
    }
    alpaka::onHost::wait(queue);
    if (calculator.failure)
      std::rethrow_exception(calculator.failure);
    if (tuner.completed() &&
        tuner.completionReason() ==
            alpakaTune::TunerCompletionReason::noValidConfiguration)
      break;
    auto const candidate = tuner.lastCandidateIndex();
    auto result = calculator.last;
    inspect(calculator, result);
    double const error = std::abs(result.estimate - std::numbers::pi);
    bool valid = false;
    double metric = error;
    if constexpr (Policy::custom) {
      // This is the application-provided objective. Submit it before the next
      // enqueue; it is independent of the post-evaluation runtime constraint.
      tuner.provideMetric(error);
      valid = result.seconds <= budgetSeconds;
    } else {
      // No provideMetric() call: this tuner uses the default timing policy.
      // Reject a measured candidate when the actual Pi result is inaccurate.
      valid = error <= maximumError;
      if (!replay)
        metric = tuner.lastConfig().runtimeSeconds.value();
    }
    outcome.trials.push_back({result, metric, valid, replay});
    if (!valid) {
      tuner.lastConfig().valid = false;
      outcome.replayCount = 0u;
      outcome.feasible = false;
    }
    if (replay && valid) {
      outcome.bestMetric = tuner.candidateMetricSamples(candidate).front();
      outcome.winner = result;
      outcome.parameters = result.parameters;
      outcome.bestError = error;
      outcome.feasible = true;
      if (++outcome.replayCount == winnerReplays)
        break;
    }
  }
  outcome.rejectedCount = tuner.info().userInvalidatedCandidateCount;
  return outcome;
}

template <typename Inspector = IgnoreInspection>
auto optimize(alpaka::onHost::concepts::Device auto device,
              alpaka::concepts::Executor auto executor,
              double budgetSeconds = 0.01,
              Objective objective = Objective::accuracy,
              double maximumError = 1.0e-3, Inspector inspect = {},
              std::string_view workload = "adaptive-pi") -> Optimization {
  if (objective == Objective::accuracy)
    return optimizeImpl(device, executor, budgetSeconds, maximumError,
                        AccuracyObjective{}, inspect, workload);
  return optimizeImpl(device, executor, budgetSeconds, maximumError,
                      RuntimeObjective{}, inspect, workload);
}
} // namespace adaptivePi
