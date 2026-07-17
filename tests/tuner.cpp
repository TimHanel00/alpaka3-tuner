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
  output << R"(schema_version: 1
tuning:
  strategy: exhaustive
  random_seed: 0
  warmup_runs: 0
  runs_per_candidate: 2
  noise_cancellation_window: 3
  max_consecutive_runs: 1
persistence:
  file: )"
         << (directory / "history.json").string() << '\n';
  return configuration;
}

} // namespace

auto main() -> int {
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

  for (std::size_t launch = 0u; launch < 6u; ++launch)
    tuner.enqueue(queue, frameSpec, bundle);
  if (!tuner.isTuningComplete() || tuner.bestCandidateIndex() >= 3u ||
      tuner.info().completionReason !=
          alpakaTune::TunerCompletionReason::allConfigurations)
    return EXIT_FAILURE;
  if (tuner.lastCandidateIndex() == std::numeric_limits<std::size_t>::max())
    return EXIT_FAILURE;

#if ALPAKA_TUNE_HAS_JSON
  auto cached = alpakaTune::makeTuner(
      alpakaTune::TunerConfig::fromYaml(configuration), tunables, device,
      alpaka::deviceKind::cpu, alpaka::api::host, executor, "tuner-test");
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
  reloadedRuntime.enqueue(queue, frameSpec, bundle);
  auto reloadedCompile =
      alpakaTune::makeTuner(alpakaTune::TunerConfig::fromYaml(configuration),
                            compileTunables, device, alpaka::deviceKind::cpu,
                            alpaka::api::host, executor, "compile-tuner-test");
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
          alpakaTune::named(numFrames,
                            alpakaTune::RVals<Index>{
                                std::vector<Index>{Index{1u}, Index{2u}}}),
          alpakaTune::named(frameExtent,
                            alpakaTune::RVals<Index>{
                                std::vector<Index>{Index{1u}, Index{2u}}})},
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
  reloadedFrame.enqueue(queue, frameSpec, frameBundle);
  if (!reloadedFrame.loadedFromCache() || !reloadedFrame.isTuningComplete())
    return EXIT_FAILURE;
#endif

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
  auto const numBlocksTunable =
      numBlocks(alpakaTune::CTypes<StaticLaunchValue, SequenceLaunchValue>{});
  auto const numThreadsTunable =
      numThreads(alpakaTune::CTypes<StaticLaunchValue, SequenceLaunchValue>{});
  auto const threadTunables = alpakaTune::constrain(
      alpakaTune::TunableBundle{numBlocksTunable, numThreadsTunable},
      alpakaTune::restrict(
          numBlocks, numThreads,
          [](alpaka::concepts::VectorOrScalar auto const &blocks,
             alpaka::concepts::VectorOrScalar auto const &threads) {
            return blocks == threads;
          }));
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

  auto budgetConfig = oneRunConfig();
  budgetConfig.maximumExecutions = 1u;
  auto budgetTuner = alpakaTune::makeTuner(budgetConfig, tunables, device,
                                           "execution-budget-test");
  budgetTuner.enqueue(queue, frameSpec, bundle);
  auto const budgetInfo = budgetTuner.info();
  if (!budgetInfo.tuningComplete || !budgetInfo.executionBudgetReached ||
      budgetInfo.executionCount != 1u || !budgetInfo.bestCandidateIndex ||
      budgetInfo.completionReason !=
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
  retiredBudgetConfig.maximumRetiredConfigurations = 2u;
  auto retiredBudgetTuner = alpakaTune::makeTuner(
      retiredBudgetConfig, tunables, device, "retired-budget-test");
  for (std::size_t launch = 0u; launch < 2u; ++launch)
    retiredBudgetTuner.enqueue(queue, frameSpec, bundle);
  auto const retiredBudgetInfo = retiredBudgetTuner.info();
  if (!retiredBudgetInfo.tuningComplete ||
      retiredBudgetInfo.executionBudgetReached ||
      retiredBudgetInfo.executionCount != 2u ||
      retiredBudgetInfo.retiredConfigurationCount != 2u ||
      retiredBudgetInfo.completionReason !=
          alpakaTune::TunerCompletionReason::maximumRetiredConfigurations)
    return EXIT_FAILURE;

#if ALPAKA_TUNE_HAS_JSON
  {
    auto history = nlohmann::json{};
    auto input = std::ifstream{configuration.parent_path() / "history.json"};
    if (!(input >> history))
      return EXIT_FAILURE;
    auto found = false;
    for (auto const &[fingerprint, cache] : history.at("contexts").items()) {
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
  reloadedRetiredBudget.enqueue(queue, frameSpec, bundle);
  if (!reloadedRetiredBudget.loadedFromCache() ||
      reloadedRetiredBudget.info().completionReason !=
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

  std::filesystem::remove_all(configuration.parent_path());
  return EXIT_SUCCESS;
}
