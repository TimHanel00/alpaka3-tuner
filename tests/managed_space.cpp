// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0
#include <alpaka/alpaka.hpp>
#include <alpakaTune/alpakaTune.hpp>
#include <alpakaTune/space/CandidateSpace.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <filesystem>
#include <fstream>
#include <set>
#include <type_traits>

#if ALPAKA_TUNE_HAS_JSON
#include <nlohmann/json.hpp>
#endif

namespace {
inline constexpr auto value = ALPAKA_TUNE_TUNABLE("value");
inline constexpr auto child = ALPAKA_TUNE_TUNABLE("child");
struct Kernel {
  ALPAKA_FN_ACC void operator()(auto const &, auto const &...) const {}
};
struct WriteIndices {
  ALPAKA_FN_ACC void operator()(auto const &acc,
                                alpaka::concepts::IMdSpan auto output,
                                alpaka::Vec<std::size_t, 1u> extent) const {
    for (auto index :
         alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid,
                                   alpaka::IdxRange{extent}))
      output[index] = static_cast<int>(index.x());
  }
};
using Vec = alpaka::Vec<std::size_t, 1u>;
auto hostDevice() {
  return alpaka::onHost::makeDeviceSelector(
             alpaka::onHost::DeviceSpec{alpaka::api::host,
                                        alpaka::deviceKind::cpu})
      .makeDevice(0u);
}
auto configFor(bool queued = false) {
  auto config = alpakaTune::TunerConfig{};
  config.mode = alpakaTune::TuningMode::onlineFixed;
  config.strategy = alpakaTune::StrategyKind::exhaustive;
  config.runsPerCandidate = config.minimumRunsPerCandidate = 1u;
  config.mannWhitneyEarlyStop = false;
  config.maximumExecutions = 1000u;
  config.maximumRetiredConfigurations.reset();
  config.queue.reset();
  if (queued)
    config.queue = alpakaTune::QueueConfig{.warmupRuns = 0u,
                                           .noiseCancellationWindow = 2u,
                                           .maxConsecutiveRuns = 1u};
  config.space.initialCandidates = 3u;
  config.space.refinementBatchSize = 3u;
  config.space.refinementInterval = 2u;
  config.space.maximumCandidates = 32u;
  config.history = {.read = false, .write = false};
  config.completeHistory = {.read = false, .write = false};
  return config;
}
} // namespace

static_assert(std::same_as<decltype(alpakaTune::domain::compileInterval<1u, 8u>(
                               alpakaTune::hint::logarithmic)),
                           alpakaTune::CVals<1u, 2u, 4u, 8u>>);
static_assert(
    decltype(alpakaTune::domain::compileInterval<1u, 1000u>())::size == 8u);

static_assert(std::same_as<
              decltype(alpakaTune::autoCandidates<1u, 8u>(
                  alpakaTune::hint::logarithmic)),
              alpakaTune::AutoCandidates<alpakaTune::CVals<1u, 2u, 4u, 8u>>>);

TEST_CASE("numeric domains keep ranges lazy and compile bounds bounded",
          "[space]") {
  auto domain = alpakaTune::domain::interval(1u, 1000000000u);
  CHECK(domain.size() == 1000000000u);
  CHECK(domain.at(999999999u) == 1000000000u);
  CHECK(domain.indexOf(99u) == 98u);
  CHECK_FALSE(domain.indexOf(0u));
  auto aligned = alpakaTune::autoCandidates(
      domain, alpakaTune::hint::alignment(32u), alpakaTune::hint::logarithmic);
  REQUIRE(aligned.project(100000u, true));
  CHECK(domain.at(*aligned.project(100000u, true)) % 32u == 0u);
  CHECK_THROWS(alpakaTune::domain::interval(8, 1));
  auto compiled =
      alpakaTune::autoCandidates(alpakaTune::domain::compileInterval<1u, 8u>(
                                     alpakaTune::hint::logarithmic),
                                 alpakaTune::hint::preferred(4u));
  CHECK(compiled.preferredIndex() == 2u);
}

TEST_CASE("a growing catalog never renumbers candidates or overcounts revisits",
          "[space]") {
  using Space = alpakaTune::detail::CandidateSpace<2u>;
  Space space;
  auto config = configFor().space;
  auto sizes = std::array<std::size_t, 2u>{10u, 10u};
  space.initialise(sizes, true, true, config, 7u);
  auto project = [](auto &, bool) { return true; };
  auto accept = [](auto const &) { return true; };
  CHECK(space.expand(project, accept) == 3u);
  auto before = space.catalog();
  CHECK_FALSE(space.observe(0u, 2.0));
  CHECK_FALSE(space.observe(0u, 1.0));
  CHECK(space.observe(1u, 1.0));
  auto best = std::array<std::size_t, 1u>{1u};
  REQUIRE(space.expand(project, accept, best) > 0u);
  for (std::size_t id{}; id < before.size(); ++id)
    CHECK(space.indices(id) == before[id]);
  CHECK(space.info().distinctMeasuredCandidateCount == 2u);
  CHECK_FALSE(space.domainExhausted());
}

TEST_CASE("generation stopping never claims unknown or stalled coverage",
          "[space]") {
  using Space = alpakaTune::detail::CandidateSpace<1u>;
  auto sizes = std::array<std::size_t, 1u>{10000u};
  auto config = configFor().space;
  config.initialCandidates = 2u;
  config.refinementBatchSize = 2u;
  config.maximumCandidates = 8u;
  config.plateauPatience = 2u;
  auto project = [](auto &, bool) { return true; };
  Space plateau;
  plateau.initialise(sizes, true, false, config, 41u);
  REQUIRE(plateau.expand(project, [](auto const &) { return true; }) == 2u);
  CHECK_FALSE(plateau.info().declaredCombinationCount);
  CHECK_FALSE(plateau.observe(0u, 1.0));
  CHECK(plateau.observe(1u, 1.0));
  auto best = std::array<std::size_t, 1u>{0u};
  REQUIRE(
      plateau.expand(project, [](auto const &) { return true; }, best) == 2u);
  CHECK_FALSE(plateau.observe(2u, 1.0));
  CHECK(plateau.state() == alpakaTune::SpaceState::plateau);
  CHECK_FALSE(plateau.domainExhausted());
  Space stalled;
  stalled.initialise(sizes, true, true, config, 41u);
  CHECK(stalled.expand(project, [](auto const &) { return false; }) == 0u);
  CHECK(stalled.state() == alpakaTune::SpaceState::stalled);
  CHECK_FALSE(stalled.domainExhausted());
}

TEST_CASE("preferred seeds do not skip finite enumeration", "[space]") {
  auto config = configFor().space;
  config.initialCandidates = 2u;
  alpakaTune::detail::CandidateSpace<1u> space;
  auto sizes = std::array<std::size_t, 1u>{8u};
  space.initialise(sizes, true, true, config, 3u);
  space.setHints({7u}, true, {false});
  auto project = [](auto &, bool) { return true; };
  auto accept = [](auto const &) { return true; };
  space.expand(project, accept);
  CHECK(space.indices(0u)[0] == 7u);
  while (space.canExpand())
    space.expand(project, accept);
  CHECK(space.domainExhausted());
  CHECK(space.size() == 8u);
  CHECK(space.find({0u}));
}

TEST_CASE("fixed scheduling refills its pool and proves finite exhaustion",
          "[space]") {
  auto device = hostDevice();
  auto queue = alpakaTune::makeQueue(device, alpaka::queueKind::nonBlocking,
                                     alpakaTune::timing::disabled);
  auto launch =
      alpaka::onHost::FrameSpec{Vec{1u}, Vec{1u}, alpaka::exec::cpuSerial};
  for (bool queued : {false, true}) {
    auto config = configFor(queued);
    auto tunables = alpakaTune::TunableBundle{
        value(alpakaTune::autoCandidates(alpakaTune::domain::interval(1, 15)))};
    auto tuner = alpakaTune::makeTuner(config, tunables, device,
                                       alpakaTune::customMetric("cost"));
    auto ids = std::set<std::size_t>{};
    for (unsigned run{}; run < 200u && !tuner.completed(); ++run) {
      tuner.enqueue(queue, launch, alpaka::KernelBundle{Kernel{}, value});
      ids.insert(tuner.lastConfig().candidateIndex);
      tuner.provideMetric(1.0 + tuner.lastConfig().candidateIndex);
    }
    CHECK(tuner.completed());
    CHECK(tuner.completionReason() ==
          alpakaTune::TunerCompletionReason::allConfigurations);
    CHECK(ids.size() == 15u);
    CHECK(tuner.info().space.domainExhausted);
    CHECK(tuner.info().space.revision > 1u);
  }
  alpaka::onHost::wait(queue);
}

TEST_CASE("every built-in strategy selects registered IDs as the catalog grows",
          "[space]") {
  auto device = hostDevice();
  auto queue = alpakaTune::makeQueue(device, alpaka::queueKind::nonBlocking,
                                     alpakaTune::timing::disabled);
  auto launch =
      alpaka::onHost::FrameSpec{Vec{1u}, Vec{1u}, alpaka::exec::cpuSerial};
  for (auto strategy :
       {alpakaTune::StrategyKind::exhaustive, alpakaTune::StrategyKind::random,
        alpakaTune::StrategyKind::simulatedAnnealing,
        alpakaTune::StrategyKind::bayesianOptimization,
        alpakaTune::StrategyKind::learnedHybrid}) {
    auto config = configFor();
    config.strategy = strategy;
    auto tunables = alpakaTune::TunableBundle{
        value(alpakaTune::autoCandidates(
            alpakaTune::domain::compileInterval<1u, 8u>(
                alpakaTune::hint::logarithmic))),
        child(alpakaTune::autoCandidates(
            alpakaTune::domain::interval(1, 1000000000)))};
    config.space.maximumCandidates = 12u;
    auto tuner = alpakaTune::makeTuner(config, tunables, device,
                                       alpakaTune::customMetric("cost"));
    for (unsigned run{}; run < 300u && !tuner.completed(); ++run) {
      tuner.enqueue(queue, launch,
                    alpaka::KernelBundle{Kernel{}, value, child});
      CHECK(tuner.lastConfig().candidateIndex < tuner.info().candidateCount);
      tuner.provideMetric(1.0 + tuner.lastConfig().candidateIndex);
    }
    CHECK(tuner.completed());
    CHECK(tuner.completionReason() ==
          alpakaTune::TunerCompletionReason::candidateBudget);
    CHECK(tuner.info().candidateCount == 12u);
  }
  alpaka::onHost::wait(queue);
}

TEST_CASE("dependent domains reject cycles and constrain child values",
          "[space]") {
  auto device = hostDevice();
  auto config = configFor();
  auto tunables = alpakaTune::TunableBundle{
      value(alpakaTune::autoCandidates(alpakaTune::domain::interval(1, 4))),
      child(alpakaTune::autoCandidates(alpakaTune::domain::dependent(
          alpakaTune::domain::interval(1, 4), value, [](int parent) {
            return alpakaTune::domain::interval(1, parent);
          })))};
  auto tuner = alpakaTune::makeTuner(config, tunables, device,
                                     alpakaTune::customMetric("cost"));
  auto queue = alpakaTune::makeQueue(device, alpaka::queueKind::nonBlocking,
                                     alpakaTune::timing::disabled);
  auto launch =
      alpaka::onHost::FrameSpec{Vec{1u}, Vec{1u}, alpaka::exec::cpuSerial};
  for (unsigned run{}; run < 100u && !tuner.completed(); ++run) {
    tuner.enqueue(queue, launch, alpaka::KernelBundle{Kernel{}, value, child});
    auto coordinates =
        tuner.candidateConfiguration(tuner.lastConfig().candidateIndex);
    CHECK(coordinates[1] <= coordinates[0]);
    tuner.provideMetric(1.0);
  }
  CHECK(tuner.completed());
  auto cyclic = alpakaTune::TunableBundle{
      value(alpakaTune::autoCandidates(alpakaTune::domain::dependent(
          alpakaTune::domain::interval(1, 4), child,
          [](int p) { return alpakaTune::domain::interval(1, p); }))),
      child(alpakaTune::autoCandidates(alpakaTune::domain::dependent(
          alpakaTune::domain::interval(1, 4), value,
          [](int p) { return alpakaTune::domain::interval(1, p); })))};
  CHECK_THROWS_AS(alpakaTune::makeTuner(config, cyclic, device,
                                        alpakaTune::customMetric("cost")),
                  std::invalid_argument);
  alpaka::onHost::wait(queue);
}

TEST_CASE("automatic type alternatives retain their compile-time identities",
          "[space]") {
  auto device = hostDevice();
  auto queue = alpakaTune::makeQueue(device, alpaka::queueKind::nonBlocking,
                                     alpakaTune::timing::disabled);
  auto launch =
      alpaka::onHost::FrameSpec{Vec{1u}, Vec{1u}, alpaka::exec::cpuSerial};
  auto tunables = alpakaTune::TunableBundle{value(alpakaTune::autoCandidates(
      alpakaTune::CTypes<int, float>{}, alpakaTune::hint::categorical))};
  auto tuner = alpakaTune::makeTuner(configFor(), tunables, device,
                                     alpakaTune::customMetric("cost"));
  for (unsigned run{}; run < 20u && !tuner.completed(); ++run) {
    tuner.enqueue(queue, launch, alpaka::KernelBundle{Kernel{}, value});
    tuner.provideMetric(1.0);
  }
  CHECK(tuner.completed());
  CHECK(tuner.info().measuredCandidateCount == 2u);
  alpaka::onHost::wait(queue);
}

TEST_CASE("adaptive revisits continue after growth reaches its budget",
          "[space]") {
  auto device = hostDevice();
  auto queue = alpakaTune::makeQueue(device, alpaka::queueKind::nonBlocking,
                                     alpakaTune::timing::disabled);
  auto launch =
      alpaka::onHost::FrameSpec{Vec{1u}, Vec{1u}, alpaka::exec::cpuSerial};
  auto config = configFor();
  config.mode = alpakaTune::TuningMode::onlineAdaptive;
  config.maximumExecutions.reset();
  config.space.maximumCandidates = 3u;
  auto tunables = alpakaTune::TunableBundle{
      value(alpakaTune::autoCandidates(alpakaTune::domain::interval(1, 100)))};
  auto tuner = alpakaTune::makeTuner(config, tunables, device,
                                     alpakaTune::customMetric("cost"));
  for (unsigned run{}; run < 20u; ++run) {
    tuner.enqueue(queue, launch, alpaka::KernelBundle{Kernel{}, value});
    tuner.provideMetric(run > 9u ? 2.0 : 1.0);
  }
  CHECK_FALSE(tuner.completed());
  CHECK(tuner.info().revisitAcceptedCount > 0u);
  CHECK(tuner.info().space.distinctMeasuredCandidateCount == 3u);
  CHECK(tuner.info().space.state == alpakaTune::SpaceState::candidateBudget);
  alpaka::onHost::wait(queue);
}

TEST_CASE("automatic geometry preserves per-axis coverage and admits overrides",
          "[space]") {
  auto device = hostDevice();
  auto queue = alpakaTune::makeQueue(device, alpaka::queueKind::nonBlocking,
                                     alpakaTune::timing::disabled);
  auto spec =
      alpaka::onHost::FrameSpec{Vec{64u}, Vec{32u}, alpaka::exec::cpuSerial};
  auto tunables =
      alpakaTune::TunableBundle{alpakaTune::makeAutomaticLaunchTuning(
          spec, alpakaTune::tuneFrameExtent(
                    spec, alpakaTune::autoCandidates(
                              alpakaTune::RVals{Vec{32u}, Vec{64u}})))};
  auto tuner = alpakaTune::makeTuner(configFor(), tunables, device,
                                     alpakaTune::customMetric("cost"));
  for (unsigned run{}; run < 100u && !tuner.completed(); ++run) {
    tuner.enqueue(queue, spec, alpaka::KernelBundle{Kernel{}});
    tuner.provideMetric(1.0);
  }
  CHECK(tuner.completed());
  CHECK(tuner.info().candidateCount == 2u);
  auto thread =
      alpaka::onHost::ThreadSpec{Vec{64u}, Vec{1u}, alpaka::exec::cpuSerial};
  auto threadTuner = alpakaTune::makeTuner(
      configFor(),
      alpakaTune::TunableBundle{alpakaTune::makeAutomaticLaunchTuning(thread)},
      device, alpakaTune::customMetric("cost"));
  threadTuner.enqueue(queue, thread, alpaka::KernelBundle{Kernel{}});
  threadTuner.provideMetric(1.0);
  CHECK(threadTuner.completed());
  CHECK(threadTuner.info().candidateCount == 1u);
  alpaka::onHost::wait(queue);
}

#if ALPAKA_TUNE_HAS_JSON
TEST_CASE(
    "generated catalogs replay offline from compact or complete histories",
    "[space][persistence]") {
  auto device = hostDevice();
  auto queue = alpakaTune::makeQueue(device, alpaka::queueKind::nonBlocking,
                                     alpakaTune::timing::disabled);
  auto launch =
      alpaka::onHost::FrameSpec{Vec{1u}, Vec{1u}, alpaka::exec::cpuSerial};
  auto tunables = alpakaTune::TunableBundle{value(
      alpakaTune::autoCandidates(alpakaTune::domain::interval(1, 1000000000)))};
  for (bool compact : {false, true}) {
    auto config = configFor();
    config.space.maximumCandidates = 9u;
    auto directory = std::filesystem::path{"managed-space-history"} /
                     (compact ? "compact" : "complete");
    std::filesystem::create_directories(directory);
    if (compact)
      config.history = {
          .file = directory / "history.json", .read = false, .write = true};
    else
      config.completeHistory = {
          .file = directory / "history.json", .read = false, .write = true};
    auto best = std::size_t{};
    {
      auto tuner = alpakaTune::makeTuner(config, tunables, device,
                                         alpakaTune::customMetric("cost"));
      for (unsigned run{}; run < 50u && !tuner.completed(); ++run) {
        tuner.enqueue(queue, launch, alpaka::KernelBundle{Kernel{}, value});
        tuner.provideMetric(10.0 - tuner.lastConfig().candidateIndex);
      }
      REQUIRE(tuner.completed());
      best = tuner.bestCandidateIndex();
      alpakaTune::flushPersistence();
    }
    config.mode = alpakaTune::TuningMode::offline;
    config.maximumExecutions.reset();
    config.history.read = compact;
    config.history.write = false;
    config.completeHistory.read = !compact;
    config.completeHistory.write = false;
    auto replay = alpakaTune::makeTuner(config, tunables, device,
                                        alpakaTune::customMetric("cost"));
    replay.enqueue(queue, launch, alpaka::KernelBundle{Kernel{}, value});
    CHECK(replay.loadedFromCache());
    CHECK(replay.bestCandidateIndex() == best);
    CHECK(replay.info().candidateCount == 9u);
    CHECK(replay.info().space.revision > 1u);
    alpaka::onHost::wait(queue);
    std::filesystem::remove_all(directory);
  }
}
#endif

#if defined(ALPAKA_CMAKE_TARGET_CUDA)
TEST_CASE("automatic launch geometry executes on CUDA with device bounds",
          "[space][cuda]") {
  auto selector = alpaka::onHost::makeDeviceSelector(alpaka::onHost::DeviceSpec{
      alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu});
  REQUIRE(selector.isAvailable());
  auto device = selector.makeDevice(0u);
  auto queue = alpakaTune::makeQueue(device, alpaka::queueKind::nonBlocking,
                                     alpakaTune::timing::enabled);
  auto spec =
      alpaka::onHost::ThreadSpec{Vec{64u}, Vec{32u}, alpaka::exec::gpuCuda};
  auto tuner = alpakaTune::makeTuner(
      configFor(),
      alpakaTune::TunableBundle{alpakaTune::makeAutomaticLaunchTuning(spec)},
      device);
  auto host = alpaka::onHost::allocHost<int>(Vec{2048u});
  auto output = alpaka::onHost::allocLike(device, host);
  for (unsigned run{}; run < 100u && !tuner.completed(); ++run) {
    alpaka::onHost::memset(queue, output, std::uint8_t{0xffu});
    tuner.enqueue(
        queue, spec,
        alpaka::KernelBundle{WriteIndices{}, output.getMdSpan(), Vec{2048u}});
    alpaka::onHost::memcpy(queue, host, output);
    alpaka::onHost::wait(queue);
    for (std::size_t index{}; index < 2048u; ++index)
      REQUIRE(host[index] == static_cast<int>(index));
  }
  CHECK(tuner.completed());
  CHECK(tuner.info().space.domainExhausted);
  CHECK(tuner.info().measuredCandidateCount > 1u);
  alpaka::onHost::wait(queue);
}
#endif
