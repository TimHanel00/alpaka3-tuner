// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>
#include <alpakaTune/alpakaTune.hpp>

#include <algorithm>
#include <atomic>
#include <concepts>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#if ALPAKA_TUNE_HAS_JSON
#include <nlohmann/json.hpp>
#endif

namespace {

inline constexpr auto runtimeValue = ALPAKA_TUNE_TUNABLE("runtimeValue");
inline constexpr auto compileTimeValue =
    ALPAKA_TUNE_TUNABLE("compileTimeValue");
inline constexpr auto compileTimeOffset =
    ALPAKA_TUNE_TUNABLE("compileTimeOffset");
inline constexpr auto numFrames = ALPAKA_TUNE_TUNABLE("numFrames");
inline constexpr auto frameExtent = ALPAKA_TUNE_TUNABLE("frameExtent");
inline constexpr auto correlatedOffset =
    ALPAKA_TUNE_TUNABLE("correlatedOffset");
inline constexpr auto numBlocks = ALPAKA_TUNE_TUNABLE("numBlocks");
inline constexpr auto numThreads = ALPAKA_TUNE_TUNABLE("numThreads");
inline constexpr auto tile = ALPAKA_TUNE_TUNABLE("tile");
inline constexpr auto workers = ALPAKA_TUNE_TUNABLE("workers");
inline constexpr auto runtimeVector = ALPAKA_TUNE_TUNABLE("runtimeVector");
inline constexpr auto compileVector = ALPAKA_TUNE_TUNABLE("compileVector");

static_assert(
    std::same_as<alpakaTune::CValsRange<0, 4, 2>, alpakaTune::CVals<0, 2, 4>>);
static_assert(
    std::same_as<alpakaTune::ParameterConfiguration, std::vector<float>>);

struct WriteKernel {
  ALPAKA_FN_ACC void operator()(auto const &acc,
                                alpaka::concepts::IMdSpan auto output,
                                int value) const {
    static_cast<void>(acc);
    output[0u] = value;
  }
};

struct CompileWriteKernel {
  ALPAKA_FN_ACC void operator()(auto const &acc,
                                alpaka::concepts::IMdSpan auto output,
                                auto value, auto offset) const {
    static_cast<void>(acc);
    output[0u] = decltype(value)::value + decltype(offset)::value;
  }
};

struct CorrelatedWriteKernel {
  ALPAKA_FN_ACC void operator()(auto const &acc,
                                alpaka::concepts::IMdSpan auto output,
                                int value, int offset) const {
    static_cast<void>(acc);
    output[0u] = value + offset;
  }
};

struct StaticFrameWriteKernel {
  ALPAKA_FN_ACC void
  operator()(auto const &acc, alpaka::concepts::IMdSpan auto output,
             alpaka::concepts::CVector auto frameExtent) const {
    static_cast<void>(acc);
    output[0u] = static_cast<int>(frameExtent[0u]);
  }
};

struct VectorAndScalarWriteKernel {
  ALPAKA_FN_ACC void operator()(auto const &acc,
                                alpaka::concepts::IMdSpan auto output,
                                alpaka::concepts::Vector auto value,
                                int scalar) const {
    static_cast<void>(acc);
    auto encoded = std::size_t{0u};
    for (std::size_t dimension = 0u; dimension < value.dim(); ++dimension)
      encoded = encoded * 100u + value[dimension];
    output[0u] = static_cast<int>(encoded * 1000u + scalar);
  }
};

auto writeConfiguration() -> std::filesystem::path {
  auto const directory =
      std::filesystem::temp_directory_path() / "alpakaTune-tuner-test";
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  auto const configuration = directory / "tuning.yaml";
  std::ofstream output{configuration};
  output << R"(schema_version: 3
tuning:
  mode: online_fixed
  strategy: exhaustive
  random_seed: 0
  warmup_runs: 0
  runs_per_candidate: 2
  noise_cancellation_window: 3
  max_consecutive_runs: 1
  maximum_executions: 100000
history:
  file: )"
         << (directory / "history.json").string() << R"(
  read: false
  write: true
complete_history:
  file: )"
         << (directory / "complete-history.json").string() << '\n';
  return configuration;
}

} // namespace

auto main() -> int {
  if (std::abs(alpakaTune::detail::normalizedLogisticAdmission(0.0, 16.0)) >
          1.0e-12 ||
      std::abs(alpakaTune::detail::normalizedLogisticAdmission(0.5, 16.0) -
               0.5) > 1.0e-12 ||
      std::abs(alpakaTune::detail::normalizedLogisticAdmission(1.0, 16.0) -
               1.0) > 1.0e-12 ||
      std::abs(alpakaTune::detail::adaptiveScoreTemperature(0.0, 0.25, 0.05) -
               0.25) > 1.0e-12 ||
      std::abs(alpakaTune::detail::adaptiveScoreTemperature(1.0, 0.25, 0.05) -
               0.05) > 1.0e-12 ||
      std::abs(alpakaTune::detail::adaptiveProgressWithActiveHistory(0.0, true,
                                                                     0.8) -
               0.8) > 1.0e-12 ||
      std::abs(alpakaTune::detail::adaptiveProgressWithActiveHistory(0.5, true,
                                                                     0.8) -
               0.9) > 1.0e-12 ||
      std::abs(alpakaTune::detail::adaptiveProgressWithActiveHistory(1.0, true,
                                                                     0.8) -
               1.0) > 1.0e-12 ||
      std::abs(alpakaTune::detail::adaptiveProgressWithActiveHistory(0.5, false,
                                                                     0.8) -
               0.5) > 1.0e-12 ||
      std::abs(alpakaTune::detail::relativeScoreAdmission(1.0, 1.0, 0.1) -
               1.0) > 1.0e-12 ||
      std::abs(alpakaTune::detail::relativeScoreAdmission(1.02, 1.0, 0.05) -
               std::exp(-0.4)) > 1.0e-12 ||
      !(alpakaTune::detail::relativeScoreAdmission(2.0, 1.0, 0.1) < 1.0))
    return EXIT_FAILURE;

  using Index = alpaka::Vec<std::size_t, 1u>;
  auto const backend =
      alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
  auto selector = alpaka::onHost::makeDeviceSelector(backend);
  if (!selector.isAvailable())
    return EXIT_SUCCESS;
  auto device = selector.makeDevice(0u);
  auto queue = device.makeQueue();
  auto const executor = alpaka::exec::cpuSerial;
  auto const frameSpec =
      alpaka::onHost::FrameSpec{Index{1u}, Index{1u}, executor};
  auto host = alpaka::onHost::allocHost<int>(Index{1u});
  auto output = alpaka::onHost::allocLike(device, host);

  auto const configuration = writeConfiguration();
  auto oneRunConfig = [&] {
    auto config = alpakaTune::TunerConfig::fromYaml(configuration);
    config.runsPerCandidate = 1u;
    config.minimumRunsPerCandidate = 1u;
    return config;
  };
  auto const tunables =
      alpakaTune::TunableBundle{runtimeValue(alpakaTune::RVals{1, 2, 3})};
  auto tuner = alpakaTune::makeTuner(
      alpakaTune::TunerConfig::fromYaml(configuration), tunables, device,
      alpaka::deviceKind::cpu, alpaka::api::host, executor, "tuner-test");
  auto const bundle =
      alpaka::KernelBundle{WriteKernel{}, output.getMdSpan(), runtimeValue};

  auto firstObservation = tuner.enqueueObserved(queue, frameSpec, bundle);
  if (!firstObservation.measured || !firstObservation.runtimeSeconds ||
      *firstObservation.runtimeSeconds < 0.0 ||
      firstObservation.candidateIndex >= 3u ||
      firstObservation.configuration.size() != 1u ||
      firstObservation.recommendationSeconds < 0.0 ||
      firstObservation.learnedStatus)
    return EXIT_FAILURE;
  auto const shortKernelWarningExpected =
      *firstObservation.runtimeSeconds < 200.0e-6;
  auto const firstTunerInfo = tuner.info();
  if (firstTunerInfo.instrumentationOverheadWarning.has_value() !=
      shortKernelWarningExpected)
    return EXIT_FAILURE;
  if (firstTunerInfo.instrumentationOverheadWarning) {
    auto const &warning = *firstTunerInfo.instrumentationOverheadWarning;
    if (warning.observedRuntimeSeconds != *firstObservation.runtimeSeconds ||
        warning.thresholdSeconds != 200.0e-6 ||
        warning.estimatedOverheadMinimumSeconds != 20.0e-6 ||
        warning.estimatedOverheadMaximumSeconds != 40.0e-6)
      return EXIT_FAILURE;
  }
  for (std::size_t launch = 1u; launch < 6u; ++launch)
    tuner.enqueue(queue, frameSpec, bundle);
  if (!tuner.isTuningComplete() || tuner.bestCandidateIndex() >= 3u ||
      tuner.completionReason() !=
          alpakaTune::TunerCompletionReason::allConfigurations)
    return EXIT_FAILURE;
  if (shortKernelWarningExpected &&
      !tuner.info().instrumentationOverheadWarning)
    return EXIT_FAILURE;
  if (tuner.lastCandidateIndex() == std::numeric_limits<std::size_t>::max())
    return EXIT_FAILURE;

#if ALPAKA_TUNE_HAS_JSON
  auto cached = alpakaTune::makeTuner(
      alpakaTune::TunerConfig::fromYaml(configuration), tunables, device,
      alpaka::deviceKind::cpu, alpaka::api::host, executor, "tuner-test");
  auto const cachedObservation =
      cached.enqueueObserved(queue, frameSpec, bundle);
  if (!cachedObservation.measured || !cachedObservation.runtimeSeconds ||
      cachedObservation.tuningComplete || !cachedObservation.loadedFromCache ||
      cached.info().executionCount != 1u ||
      cached.info().measuredCandidateCount != 3u ||
      cached.candidateRuntimeSamples(cachedObservation.candidateIndex).size() !=
          3u)
    return EXIT_FAILURE;
  for (std::size_t launch = 1u; launch < 6u; ++launch)
    cached.enqueue(queue, frameSpec, bundle);
  alpaka::onHost::wait(queue);
  if (!cached.loadedFromCache() || !cached.isTuningComplete())
    return EXIT_FAILURE;
  alpaka::onHost::memcpy(queue, host, output);
  alpaka::onHost::wait(queue);
  if (host[0u] < 1 || host[0u] > 3)
    return EXIT_FAILURE;
#endif

  auto const compileTimeValueTunable =
      compileTimeValue(alpakaTune::CVals<1, 2>{});
  auto const compileTimeOffsetTunable =
      compileTimeOffset(alpakaTune::CVals<4, 8>{});
  auto const compileTunables = alpakaTune::TunableBundle{
      compileTimeValueTunable, compileTimeOffsetTunable};
  auto compileTuner =
      alpakaTune::makeTuner(alpakaTune::TunerConfig::fromYaml(configuration),
                            compileTunables, device, alpaka::deviceKind::cpu,
                            alpaka::api::host, executor, "compile-tuner-test");
  auto const compileBundle =
      alpaka::KernelBundle{CompileWriteKernel{}, output.getMdSpan(),
                           alpakaTune::markTunable(compileTimeValue),
                           alpakaTune::markTunable(compileTimeOffset)};
  for (std::size_t launch = 0u; launch < 8u; ++launch)
    compileTuner.enqueue(queue, frameSpec, compileBundle);
  if (!compileTuner.isTuningComplete() ||
      compileTuner.bestCandidateIndex() >= 4u)
    return EXIT_FAILURE;

#if ALPAKA_TUNE_HAS_JSON
  // Both tuners share one persistence file. Writing the compile-time tuner
  // must preserve the previously completed runtime tuner.
  auto reloadedRuntime = alpakaTune::makeTuner(
      alpakaTune::TunerConfig::fromYaml(configuration), tunables, device,
      alpaka::deviceKind::cpu, alpaka::api::host, executor, "tuner-test");
  for (std::size_t launch = 0u; launch < 6u; ++launch)
    reloadedRuntime.enqueue(queue, frameSpec, bundle);
  auto reloadedCompile =
      alpakaTune::makeTuner(alpakaTune::TunerConfig::fromYaml(configuration),
                            compileTunables, device, alpaka::deviceKind::cpu,
                            alpaka::api::host, executor, "compile-tuner-test");
  for (std::size_t launch = 0u; launch < 8u; ++launch)
    reloadedCompile.enqueue(queue, frameSpec, compileBundle);
  alpaka::onHost::wait(queue);
  if (!reloadedRuntime.loadedFromCache() ||
      !reloadedRuntime.isTuningComplete() ||
      !reloadedCompile.loadedFromCache() || !reloadedCompile.isTuningComplete())
    return EXIT_FAILURE;
#endif

  // Reserved launch dimensions remain host-side: no KernelBundle marker is
  // required, yet the tuner rebuilds the FrameSpec for each candidate.
  auto const frameTunables = alpakaTune::constrain(
      alpakaTune::TunableBundle{
          alpakaTune::tuneNumFrames(frameSpec,
                                    alpakaTune::RVals<Index>{std::vector<Index>{
                                        Index{1u}, Index{2u}}}),
          alpakaTune::tuneFrameExtent(
              frameSpec, alpakaTune::RVals<Index>{std::vector<Index>{
                             Index{1u}, Index{2u}}})},
      alpakaTune::restrict(
          numFrames, frameExtent,
          [](alpaka::concepts::VectorOrScalar auto const &frames,
             alpaka::concepts::VectorOrScalar auto const &extent) {
            return frames == extent;
          }));
  auto frameTuner = alpakaTune::makeTuner(
      alpakaTune::TunerConfig::fromYaml(configuration), frameTunables, device,
      alpaka::deviceKind::cpu, alpaka::api::host, executor, "frame-tuner-test");
  auto const frameBundle =
      alpaka::KernelBundle{WriteKernel{}, output.getMdSpan(), 7};
  for (std::size_t launch = 0u; launch < 4u; ++launch)
    frameTuner.enqueue(queue, frameSpec, frameBundle);
  alpaka::onHost::wait(queue);
  if (!frameTuner.isTuningComplete() || frameTuner.bestCandidateIndex() >= 4u)
    return EXIT_FAILURE;

#if ALPAKA_TUNE_HAS_JSON
  auto reloadedFrame = alpakaTune::makeTuner(
      alpakaTune::TunerConfig::fromYaml(configuration), frameTunables, device,
      alpaka::deviceKind::cpu, alpaka::api::host, executor, "frame-tuner-test");
  for (std::size_t launch = 0u; launch < 4u; ++launch)
    reloadedFrame.enqueue(queue, frameSpec, frameBundle);
  if (!reloadedFrame.loadedFromCache() || !reloadedFrame.isTuningComplete())
    return EXIT_FAILURE;
#endif

  // A correlated launch fragment flattens into the same TunableBundle as an
  // ordinary kernel parameter. The default factory supplies both launch
  // entries and their coverage-preserving relation.
  auto const defaultFrameSpec =
      alpaka::onHost::FrameSpec{Index{1u}, Index{2u}, executor};
  auto const defaultFrameTunables = alpakaTune::TunableBundle{
      alpakaTune::makeFrameSpecTuning(defaultFrameSpec),
      runtimeValue(alpakaTune::RVals{1, 2})};
  static_assert(std::remove_cvref_t<decltype(defaultFrameTunables)>::size ==
                3u);
  auto defaultFrameTuner =
      alpakaTune::makeTuner(oneRunConfig(), defaultFrameTunables, device,
                            executor, "default-frame-fragment-test");
  for (std::size_t launch = 0u; launch < 4u; ++launch)
    defaultFrameTuner.enqueue(queue, defaultFrameSpec, bundle);
  auto const defaultFrameInfo = defaultFrameTuner.info();
  if (!defaultFrameInfo.tuningComplete ||
      defaultFrameInfo.candidateCount != 8u ||
      defaultFrameInfo.rejectedCandidateCount != 4u ||
      defaultFrameInfo.measuredCandidateCount != 4u)
    return EXIT_FAILURE;

  using StaticFrameExtent = alpaka::CVec<std::size_t, 4u>;
  using AlternateStaticFrameExtent = alpaka::CVec<std::size_t, 2u>;
  auto const staticFrameSpec =
      alpaka::onHost::FrameSpec{Index{1u}, StaticFrameExtent{}, executor};
  auto const staticFrameTunables = alpakaTune::constrain(
      alpakaTune::TunableBundle{
          alpakaTune::named(numFrames,
                            alpakaTune::RVals<Index>{
                                std::vector<Index>{Index{1u}, Index{2u}}}),
          alpakaTune::named(frameExtent,
                            alpakaTune::CTypes<StaticFrameExtent,
                                               AlternateStaticFrameExtent>{})},
      alpakaTune::restrict(
          numFrames, frameExtent,
          [](alpaka::concepts::VectorOrScalar auto const &frames,
             alpaka::concepts::VectorOrScalar auto const &extent) {
            return frames[0u] * extent[0u] == 4u;
          }));
  auto staticFrameTuner = alpakaTune::makeTuner(
      alpakaTune::TunerConfig::fromYaml(configuration), staticFrameTunables,
      device, alpaka::deviceKind::cpu, alpaka::api::host, executor,
      "static-frame-tuner-test");
  auto const staticFrameBundle =
      alpaka::KernelBundle{StaticFrameWriteKernel{}, output.getMdSpan(),
                           alpakaTune::markTunable(frameExtent)};
  for (std::size_t launch = 0u; launch < 4u; ++launch)
    staticFrameTuner.enqueue(queue, staticFrameSpec, staticFrameBundle);
  alpaka::onHost::wait(queue);
  if (!staticFrameTuner.isTuningComplete() ||
      staticFrameTuner.bestCandidateIndex() >= 4u)
    return EXIT_FAILURE;

  auto const correlatedTunables = alpakaTune::constrain(
      alpakaTune::TunableBundle{
          alpakaTune::named(runtimeValue, alpakaTune::RVals{1, 2, 3}),
          alpakaTune::named(correlatedOffset, alpakaTune::RVals{10, 20, 30})},
      alpakaTune::restrict(
          runtimeValue, correlatedOffset,
          [](alpaka::concepts::VectorOrScalar auto const &value,
             alpaka::concepts::VectorOrScalar auto const &offset) {
            return value * 10 == offset;
          }));
  auto correlatedTuner = alpakaTune::makeTuner(
      alpakaTune::TunerConfig::fromYaml(configuration), correlatedTunables,
      device, alpaka::deviceKind::cpu, alpaka::api::host, executor,
      "correlated-tuner-test");
  auto const correlatedBundle =
      alpaka::KernelBundle{CorrelatedWriteKernel{}, output.getMdSpan(),
                           alpakaTune::markTunable(runtimeValue),
                           alpakaTune::markTunable(correlatedOffset)};
  for (std::size_t launch = 0u; launch < 6u; ++launch)
    correlatedTuner.enqueue(queue, frameSpec, correlatedBundle);
  if (!correlatedTuner.isTuningComplete() ||
      correlatedTuner.bestCandidateIndex() >= 9u)
    return EXIT_FAILURE;

  using StaticLaunchValue = alpaka::CVec<std::size_t, 1u>;
  using SequenceLaunchValue = std::integer_sequence<std::size_t, 1u>;
  auto const threadSpec = alpaka::onHost::ThreadSpec{
      StaticLaunchValue{}, StaticLaunchValue{}, executor};
  auto const numBlocksTunable = alpakaTune::tuneNumBlocks(
      threadSpec, alpakaTune::CTypes<StaticLaunchValue, SequenceLaunchValue>{});
  auto const numThreadsTunable = alpakaTune::tuneNumThreads(
      threadSpec, alpakaTune::CTypes<StaticLaunchValue, SequenceLaunchValue>{});
  auto const threadTunables =
      alpakaTune::TunableBundle{alpakaTune::makeThreadSpecTuning(
          numBlocksTunable, numThreadsTunable,
          alpakaTune::restrict(
              numBlocks, numThreads,
              [](alpaka::concepts::VectorOrScalar auto const &blocks,
                 alpaka::concepts::VectorOrScalar auto const &threads) {
                return blocks == threads;
              }))};
  auto threadTuner =
      alpakaTune::makeTuner(alpakaTune::TunerConfig::fromYaml(configuration),
                            threadTunables, device, alpaka::deviceKind::cpu,
                            alpaka::api::host, executor, "thread-tuner-test");
  for (std::size_t launch = 0u; launch < 8u; ++launch)
    threadTuner.enqueue(queue, threadSpec, frameBundle);
  if (!threadTuner.isTuningComplete() || threadTuner.bestCandidateIndex() >= 4u)
    return EXIT_FAILURE;

  auto restrictionCalls = std::atomic<std::size_t>{};
  auto const lazyTunables = alpakaTune::constrain(
      alpakaTune::TunableBundle{
          alpakaTune::named(tile, alpakaTune::RVals{1, 2, 3}),
          alpakaTune::named(workers, alpakaTune::RVals{1, 2, 3})},
      alpakaTune::restrict(
          workers, tile,
          [&restrictionCalls](
              alpaka::concepts::VectorOrScalar auto const &workerCount,
              alpaka::concepts::VectorOrScalar auto const &tileExtent) {
            restrictionCalls.fetch_add(1u);
            return workerCount <= tileExtent;
          }));
  auto lazyTuner = alpakaTune::makeTuner(oneRunConfig(), lazyTunables, device,
                                         "lazy-restriction-test");
  if (restrictionCalls.load() != 0u)
    return EXIT_FAILURE;
  auto const lazyBundle = alpaka::KernelBundle{
      CorrelatedWriteKernel{}, output.getMdSpan(),
      alpakaTune::markTunable(tile), alpakaTune::markTunable(workers)};
  for (std::size_t launch = 0u; launch < 6u; ++launch)
    lazyTuner.enqueue(queue, frameSpec, lazyBundle);
  auto const lazyInfo = lazyTuner.info();
  if (!lazyInfo.tuningComplete || lazyInfo.candidateCount != 9u ||
      lazyInfo.rejectedCandidateCount != 3u ||
      lazyInfo.measuredCandidateCount != 6u || restrictionCalls.load() != 9u)
    return EXIT_FAILURE;

  auto retryLimitConfig = oneRunConfig();
  retryLimitConfig.maximumConsecutiveStrategyRetries = 3u;
  retryLimitConfig.noiseCancellationWindow = 1u;
  retryLimitConfig.history.file.reset();
  retryLimitConfig.history.read = false;
  retryLimitConfig.history.write = false;
  retryLimitConfig.completeHistory.file.reset();
  retryLimitConfig.completeHistory.read = false;
  retryLimitConfig.completeHistory.write = false;
  auto const rejectedTunables = alpakaTune::constrain(
      alpakaTune::TunableBundle{
          alpakaTune::named(runtimeValue, alpakaTune::RVals{1, 2, 3, 4, 5, 6})},
      alpakaTune::restrict(
          runtimeValue,
          [](alpaka::concepts::VectorOrScalar auto const &) { return false; }));
  auto retryLimitTuner = alpakaTune::makeTuner(
      retryLimitConfig, rejectedTunables, device, "strategy-retry-limit-test");
  auto retryLimitThrew = false;
  try {
    retryLimitTuner.enqueue(queue, frameSpec, bundle);
  } catch (std::invalid_argument const &) {
    retryLimitThrew = true;
  }
  auto const retryLimitInfo = retryLimitTuner.info();
  if (!retryLimitThrew || !retryLimitTuner.isTuningComplete() ||
      !retryLimitTuner.completed() ||
      retryLimitTuner.completionReason() !=
          alpakaTune::TunerCompletionReason::
              maximumConsecutiveStrategyRetries ||
      retryLimitInfo.maximumConsecutiveStrategyRetries != 3u ||
      retryLimitInfo.consecutiveStrategyRetries != 3u ||
      retryLimitInfo.strategyRetryLimitReachedCount != 1u ||
      retryLimitInfo.adaptiveRetryFallbackCount != 0u ||
      retryLimitInfo.restrictionRejectedCount != 3u ||
      retryLimitInfo.bestCandidateIndex || !retryLimitInfo.completionReason ||
      *retryLimitInfo.completionReason !=
          alpakaTune::TunerCompletionReason::maximumConsecutiveStrategyRetries)
    return EXIT_FAILURE;

  auto adaptiveRetryConfig = retryLimitConfig;
  adaptiveRetryConfig.mode = alpakaTune::TuningMode::onlineAdaptive;
  adaptiveRetryConfig.maximumExecutions.reset();
  adaptiveRetryConfig.maximumRetiredConfigurations.reset();
  adaptiveRetryConfig.maximumConsecutiveStrategyRetries = 1u;
  adaptiveRetryConfig.horizon = std::numeric_limits<std::size_t>::max();
  auto emptyAdaptiveRetryTuner =
      alpakaTune::makeTuner(adaptiveRetryConfig, rejectedTunables, device,
                            "strategy-retry-empty-adaptive-test");
  auto emptyAdaptiveRetryThrew = false;
  try {
    emptyAdaptiveRetryTuner.enqueue(queue, frameSpec, bundle);
  } catch (std::invalid_argument const &) {
    emptyAdaptiveRetryThrew = true;
  }
  if (!emptyAdaptiveRetryThrew || emptyAdaptiveRetryTuner.isTuningComplete() ||
      emptyAdaptiveRetryTuner.completed() ||
      emptyAdaptiveRetryTuner.info().strategyRetryLimitReachedCount != 1u ||
      emptyAdaptiveRetryTuner.info().adaptiveRetryFallbackCount != 0u ||
      emptyAdaptiveRetryTuner.info().bestCandidateIndex)
    return EXIT_FAILURE;

  auto const singleCandidateTunables = alpakaTune::TunableBundle{
      alpakaTune::named(runtimeValue, alpakaTune::RVals{7})};
  auto adaptiveRetryTuner =
      alpakaTune::makeTuner(adaptiveRetryConfig, singleCandidateTunables,
                            device, "strategy-retry-adaptive-fallback-test");
  auto const firstAdaptiveRetry =
      adaptiveRetryTuner.enqueueObserved(queue, frameSpec, bundle);
  if (!firstAdaptiveRetry.measured || firstAdaptiveRetry.tuningComplete ||
      adaptiveRetryTuner.isTuningComplete() ||
      adaptiveRetryTuner.info().strategyRetryLimitReachedCount != 1u ||
      adaptiveRetryTuner.info().adaptiveRetryFallbackCount != 1u)
    return EXIT_FAILURE;
  auto const secondAdaptiveRetry =
      adaptiveRetryTuner.enqueueObserved(queue, frameSpec, bundle);
  if (!secondAdaptiveRetry.measured || !secondAdaptiveRetry.runtimeSeconds ||
      secondAdaptiveRetry.tuningComplete ||
      secondAdaptiveRetry.candidateIndex != firstAdaptiveRetry.candidateIndex ||
      adaptiveRetryTuner.isTuningComplete() ||
      adaptiveRetryTuner.info().strategyRetryLimitReachedCount != 2u ||
      adaptiveRetryTuner.info().adaptiveRetryFallbackCount != 2u ||
      adaptiveRetryTuner.candidateRuntimeSamples(0u).size() != 2u)
    return EXIT_FAILURE;
  auto adaptiveRetryReasonRejected = false;
  try {
    static_cast<void>(adaptiveRetryTuner.completionReason());
  } catch (std::logic_error const &) {
    adaptiveRetryReasonRejected = true;
  }
  if (!adaptiveRetryReasonRejected)
    return EXIT_FAILURE;

  auto budgetConfig = oneRunConfig();
  budgetConfig.maximumExecutions = 1u;
  auto budgetTuner = alpakaTune::makeTuner(budgetConfig, tunables, device,
                                           "execution-budget-test");
  auto const budgetObservation =
      budgetTuner.enqueueObserved(queue, frameSpec, bundle);
  auto const budgetInfo = budgetTuner.info();
  if (!budgetObservation.tuningComplete || !budgetInfo.tuningComplete ||
      !budgetTuner.isTuningComplete() || !budgetTuner.completed() ||
      !budgetInfo.executionBudgetReached || budgetInfo.executionCount != 1u ||
      !budgetInfo.bestCandidateIndex ||
      budgetTuner.completionReason() !=
          alpakaTune::TunerCompletionReason::maximumExecutions)
    return EXIT_FAILURE;

#if ALPAKA_TUNE_HAS_JSON
  auto reloadedBudget = alpakaTune::makeTuner(budgetConfig, tunables, device,
                                              "execution-budget-test");
  reloadedBudget.enqueue(queue, frameSpec, bundle);
  if (!reloadedBudget.loadedFromCache() ||
      !reloadedBudget.info().executionBudgetReached)
    return EXIT_FAILURE;
#endif

  auto retiredBudgetConfig = oneRunConfig();
  retiredBudgetConfig.maximumExecutions.reset();
  retiredBudgetConfig.maximumRetiredConfigurations = 2u;
  auto retiredBudgetTuner = alpakaTune::makeTuner(
      retiredBudgetConfig, tunables, device, "retired-budget-test");
  while (!retiredBudgetTuner.completed())
    retiredBudgetTuner.enqueue(queue, frameSpec, bundle);
  auto const retiredBudgetInfo = retiredBudgetTuner.info();
  if (!retiredBudgetInfo.tuningComplete || !retiredBudgetTuner.completed() ||
      retiredBudgetInfo.executionBudgetReached ||
      retiredBudgetInfo.executionCount != 2u ||
      retiredBudgetInfo.retiredConfigurationCount != 2u ||
      retiredBudgetTuner.completionReason() !=
          alpakaTune::TunerCompletionReason::maximumRetiredConfigurations)
    return EXIT_FAILURE;

  // Adaptive mode ends each queue residency after one configured activation
  // burst. Warm-ups remain queue-controlled and do not enter the rolling
  // timing window: 4 consecutive launches - 1 warm-up = 3 samples.
  auto adaptiveConfig = oneRunConfig();
  adaptiveConfig.mode = alpakaTune::TuningMode::onlineAdaptive;
  adaptiveConfig.maximumExecutions.reset();
  adaptiveConfig.maximumRetiredConfigurations.reset();
  adaptiveConfig.horizon = 4u;
  adaptiveConfig.warmupRuns = 1u;
  adaptiveConfig.maxConsecutiveRuns = 4u;
  adaptiveConfig.noiseCancellationWindow = 1u;
  adaptiveConfig.historyWindowSize = 3u;
  auto const adaptiveTunables =
      alpakaTune::TunableBundle{runtimeValue(alpakaTune::RVals{7})};
  auto adaptiveTuner = alpakaTune::makeTuner(adaptiveConfig, adaptiveTunables,
                                             device, "adaptive-burst-test");
  if (adaptiveTuner.completed())
    return EXIT_FAILURE;
  for (std::size_t launch = 0u; launch < 4u; ++launch)
    adaptiveTuner.enqueue(queue, frameSpec, bundle);
  auto const firstAdaptiveInfo = adaptiveTuner.info();
  if (!firstAdaptiveInfo.tuningComplete || !adaptiveTuner.completed() ||
      !adaptiveTuner.isTuningComplete() ||
      firstAdaptiveInfo.executionCount != 4u ||
      firstAdaptiveInfo.retiredConfigurationCount != 1u ||
      adaptiveTuner.candidateRuntimeSamples(0u).size() != 3u)
    return EXIT_FAILURE;
  auto adaptiveReasonRejected = false;
  try {
    static_cast<void>(adaptiveTuner.completionReason());
  } catch (std::logic_error const &) {
    adaptiveReasonRejected = true;
  }
  if (!adaptiveReasonRejected)
    return EXIT_FAILURE;

  // horizon is an adaptive schedule, not a stop.
  // At and after the horizon the single current-best candidate passes both
  // admission gates with probability one and may be visited indefinitely.
  auto const postHorizonObservation =
      adaptiveTuner.enqueueObserved(queue, frameSpec, bundle);
  if (!postHorizonObservation.measured ||
      !postHorizonObservation.runtimeSeconds ||
      !postHorizonObservation.tuningComplete)
    return EXIT_FAILURE;
  for (std::size_t launch = 1u; launch < 4u; ++launch)
    adaptiveTuner.enqueue(queue, frameSpec, bundle);
  auto const secondAdaptiveInfo = adaptiveTuner.info();
  if (!secondAdaptiveInfo.tuningComplete || !adaptiveTuner.completed() ||
      !adaptiveTuner.isTuningComplete() ||
      secondAdaptiveInfo.executionCount != 8u ||
      secondAdaptiveInfo.retiredConfigurationCount != 2u ||
      secondAdaptiveInfo.revisitAcceptedCount == 0u ||
      adaptiveTuner.candidateRuntimeSamples(0u).size() != 3u)
    return EXIT_FAILURE;

  // A compatible adaptive history starts the sigmoid/Boltzmann schedule at
  // the configured offset, but still receives a complete new-run horizon.
  auto resumedAdaptiveTuner = alpakaTune::makeTuner(
      adaptiveConfig, adaptiveTunables, device, "adaptive-burst-test");
  for (std::size_t launch = 0u; launch < 3u; ++launch) {
    resumedAdaptiveTuner.enqueue(queue, frameSpec, bundle);
    if (resumedAdaptiveTuner.completed())
      return EXIT_FAILURE;
  }
  resumedAdaptiveTuner.enqueue(queue, frameSpec, bundle);
  if (!resumedAdaptiveTuner.loadedFromCache() ||
      !resumedAdaptiveTuner.completed() ||
      resumedAdaptiveTuner.info().executionCount != 4u ||
      resumedAdaptiveTuner.info().adaptiveHorizonExecutionCount != 4u ||
      std::abs(resumedAdaptiveTuner.info().adaptiveHorizonProgress - 1.0) >
          1.0e-12)
    return EXIT_FAILURE;

#if ALPAKA_TUNE_HAS_JSON
  // Offline mode accepts the compatible adaptive history even though it has
  // no terminal completion reason, launches its recorded best, and does not
  // collect a timing sample or instantiate a strategy.
  auto offlineConfig = adaptiveConfig;
  offlineConfig.mode = alpakaTune::TuningMode::offline;
  auto offlineTuner = alpakaTune::makeTuner(offlineConfig, adaptiveTunables,
                                            device, "adaptive-burst-test");
  auto const offlineObservation =
      offlineTuner.enqueueObserved(queue, frameSpec, bundle);
  if (!offlineTuner.loadedFromCache() || !offlineTuner.isTuningComplete() ||
      !offlineTuner.completed() ||
      offlineTuner.completionReason() !=
          alpakaTune::TunerCompletionReason::offlineReplay ||
      !offlineObservation.tuningComplete ||
      !offlineTuner.info().tuningComplete || offlineObservation.measured ||
      offlineObservation.runtimeSeconds ||
      offlineTuner.info().mode != alpakaTune::TuningMode::offline ||
      offlineTuner.candidateRuntimeSamples(0u).size() != 3u)
    return EXIT_FAILURE;

  auto const oldHistoryPath = configuration.parent_path() / "history-v10.json";
  {
    auto output = std::ofstream{oldHistoryPath};
    output << R"({"schema_version":10,"contexts":{}})";
  }
  auto oldHistoryConfig = offlineConfig;
  oldHistoryConfig.completeHistory.file = oldHistoryPath;
  auto oldHistoryRejected = false;
  try {
    auto oldHistoryTuner = alpakaTune::makeTuner(
        oldHistoryConfig, adaptiveTunables, device, "old-history-test");
    oldHistoryTuner.enqueue(queue, frameSpec, bundle);
  } catch (std::runtime_error const &error) {
    oldHistoryRejected = std::string_view{error.what()}.find(
                             "fresh histories") != std::string_view::npos;
  }
  if (!oldHistoryRejected)
    return EXIT_FAILURE;
#endif

#if ALPAKA_TUNE_HAS_JSON
  {
    auto const historyPath =
        configuration.parent_path() / "complete-history.json";
    if (std::filesystem::exists(historyPath))
      return EXIT_FAILURE;
    auto const contexts =
        alpakaTune::detail::completeHistoryStore(historyPath)
            ->stagedCaches(
                decltype(retiredBudgetTuner)::completeHistorySchemaVersion);
    auto found = false;
    for (auto const &[fingerprint, cache] : contexts) {
      static_cast<void>(fingerprint);
      auto const identities = cache.at("metadata")
                                  .at("identity_entries")
                                  .get<std::vector<std::string>>();
      if (std::none_of(identities.begin(), identities.end(),
                       [](std::string const &entry) {
                         return entry.ends_with("=retired-budget-test");
                       }))
        continue;
      found = true;
      if (cache.at("completion_reason") != "maximum_retired_configurations" ||
          cache.at("candidate_count") != 3u ||
          cache.at("retired_configuration_count") != 2u ||
          cache.at("best_improvements").empty() ||
          cache.at("candidate_estimates").size() != 3u ||
          cache.at("candidate_configurations").size() != 3u ||
          cache.at("candidate_configurations").at(0).at("runtimeValue") !=
              "1" ||
          std::count_if(
              cache.at("candidate_estimates").begin(),
              cache.at("candidate_estimates").end(),
              [](auto const &estimate) { return !estimate.is_null(); }) != 2)
        return EXIT_FAILURE;
      auto previousRuntime = std::numeric_limits<double>::infinity();
      for (auto const &improvement : cache.at("best_improvements")) {
        auto const runtime = improvement.at("runtime_seconds").get<double>();
        if (runtime >= previousRuntime ||
            improvement.at("execution_count").get<std::size_t>() > 2u)
          return EXIT_FAILURE;
        previousRuntime = runtime;
      }
    }
    if (!found)
      return EXIT_FAILURE;
  }

  auto reloadedRetiredBudget = alpakaTune::makeTuner(
      retiredBudgetConfig, tunables, device, "retired-budget-test");
  while (!reloadedRetiredBudget.completed())
    reloadedRetiredBudget.enqueue(queue, frameSpec, bundle);
  if (!reloadedRetiredBudget.loadedFromCache() ||
      reloadedRetiredBudget.info().executionCount != 2u ||
      reloadedRetiredBudget.completionReason() !=
          alpakaTune::TunerCompletionReason::maximumRetiredConfigurations)
    return EXIT_FAILURE;
#endif

  if (alpakaTune::generate::linSpace(1, 3, 1).size() != 3u ||
      alpakaTune::generate::logSpace(1, 4, 2).size() != 3u ||
      alpakaTune::generate::linSpace(Index{1u}, Index{3u}, Index{1u}).size() !=
          3u)
    return EXIT_FAILURE;

  using Vector3 = alpaka::Vec<std::size_t, 3u>;
  auto const runtimeVectorTunables = alpakaTune::TunableBundle{
      runtimeVector(alpakaTune::generate::linSpace(
          Vector3{1u, 11u, 21u}, Vector3{10u, 20u, 30u}, Vector3::fill(1u))),
      workers(alpakaTune::RVals{1, 2})};
  auto runtimeVectorTuner =
      alpakaTune::makeTuner(oneRunConfig(), runtimeVectorTunables, device,
                            "runtime-vector-dimensions-test");
  if (runtimeVectorTuner.candidateConfiguration(0u).size() != 4u ||
      runtimeVectorTuner.info().candidateCount != 2000u)
    return EXIT_FAILURE;
  auto const runtimeVectorBundle = alpaka::KernelBundle{
      VectorAndScalarWriteKernel{}, output.getMdSpan(), runtimeVector, workers};
  auto runtimeResults = std::set<int>{};
  for (std::size_t launch = 0u; launch < 2000u; ++launch) {
    runtimeVectorTuner.enqueue(queue, frameSpec, runtimeVectorBundle);
    alpaka::onHost::memcpy(queue, host, output);
    alpaka::onHost::wait(queue);
    runtimeResults.insert(host[0u]);
  }
  if (!runtimeVectorTuner.isTuningComplete() || runtimeResults.size() != 2000u)
    return EXIT_FAILURE;
  auto runtimeVectorOfflineConfig = oneRunConfig();
  runtimeVectorOfflineConfig.mode = alpakaTune::TuningMode::offline;
  runtimeVectorOfflineConfig.maximumExecutions.reset();
  runtimeVectorOfflineConfig.maximumRetiredConfigurations.reset();
  auto runtimeVectorOffline =
      alpakaTune::makeTuner(runtimeVectorOfflineConfig, runtimeVectorTunables,
                            device, "runtime-vector-dimensions-test");
  runtimeVectorOffline.enqueue(queue, frameSpec, runtimeVectorBundle);
  if (!runtimeVectorOffline.loadedFromCache() ||
      !runtimeVectorOffline.isTuningComplete())
    return EXIT_FAILURE;

  using CompileVectorA = alpaka::CVec<std::size_t, 1u, 10u>;
  using CompileVectorB = std::integer_sequence<std::size_t, 2u, 20u>;
  auto const compileVectorTunables = alpakaTune::TunableBundle{
      compileVector(alpakaTune::CTypes<CompileVectorA, CompileVectorB>{}),
      workers(alpakaTune::generate::linSpace(1, 500, 1))};
  auto compileVectorTuner =
      alpakaTune::makeTuner(oneRunConfig(), compileVectorTunables, device,
                            "compile-vector-dimensions-test");
  if (compileVectorTuner.candidateConfiguration(0u).size() != 3u ||
      compileVectorTuner.info().candidateCount != 2000u)
    return EXIT_FAILURE;
  auto const compileVectorBundle = alpaka::KernelBundle{
      VectorAndScalarWriteKernel{}, output.getMdSpan(), compileVector, workers};
  auto compileResults = std::set<int>{};
  for (std::size_t launch = 0u; launch < 2000u; ++launch) {
    compileVectorTuner.enqueue(queue, frameSpec, compileVectorBundle);
    alpaka::onHost::memcpy(queue, host, output);
    alpaka::onHost::wait(queue);
    compileResults.insert(host[0u]);
  }
  if (!compileVectorTuner.isTuningComplete() || compileResults.size() != 2000u)
    return EXIT_FAILURE;
  auto compileVectorOfflineConfig = oneRunConfig();
  compileVectorOfflineConfig.mode = alpakaTune::TuningMode::offline;
  compileVectorOfflineConfig.maximumExecutions.reset();
  compileVectorOfflineConfig.maximumRetiredConfigurations.reset();
  auto compileVectorOffline =
      alpakaTune::makeTuner(compileVectorOfflineConfig, compileVectorTunables,
                            device, "compile-vector-dimensions-test");
  compileVectorOffline.enqueue(queue, frameSpec, compileVectorBundle);
  if (!compileVectorOffline.loadedFromCache() ||
      !compileVectorOffline.isTuningComplete())
    return EXIT_FAILURE;

#if ALPAKA_TUNE_HAS_JSON
  alpakaTune::flushPersistence();
  auto compactHistory = nlohmann::json{};
  auto compactHistoryInput =
      std::ifstream{configuration.parent_path() / "history.json"};
  compactHistoryInput >> compactHistory;
  if (compactHistory.at("schema_version") !=
          alpakaTune::detail::historySchemaVersion ||
      compactHistory.at("contexts").empty())
    return EXIT_FAILURE;
  for (auto const &[fingerprint, context] :
       compactHistory.at("contexts").items()) {
    static_cast<void>(fingerprint);
    if (context.at("configurations").empty())
      return EXIT_FAILURE;
  }
#endif

  std::filesystem::remove_all(configuration.parent_path());
  return EXIT_SUCCESS;
}
