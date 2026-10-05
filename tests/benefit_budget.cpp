// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0
#include <alpakaTune/alpakaTune.hpp>
#include <alpakaTune/core/peripherals/BenefitBudget.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>

namespace {
inline constexpr auto choice = ALPAKA_TUNE_TUNABLE("budgetChoice");
struct Kernel {
  ALPAKA_FN_ACC void operator()(auto const &,
                                alpaka::concepts::IMdSpan auto output,
                                auto value) const {
    ++output[0u];
    output[1u] = static_cast<int>(value);
  }
};
using Vec = alpaka::Vec<std::size_t, 1u>;
auto device() {
  return alpaka::onHost::makeDeviceSelector(
             alpaka::onHost::DeviceSpec{alpaka::api::host,
                                        alpaka::deviceKind::cpu})
      .makeDevice(0u);
}
auto config() {
  alpakaTune::TunerConfig result;
  result.budget = alpakaTune::BudgetConfig{.expectedLaunches = 10000u};
  result.history = {.read = false, .write = false};
  result.completeHistory = {.read = false, .write = false};
  return result;
}
} // namespace

TEST_CASE("economic admission depends on future savings and costs",
          "[budget]") {
  using alpakaTune::detail::BenefitBudget;
  auto policy = alpakaTune::BudgetConfig{.expectedLaunches = 100u};
  BenefitBudget budget(policy);
  budget.establishBaseline(0.001);
  CHECK(budget.ceiling() == 0.005);
  CHECK_FALSE(budget.worthwhile(0.0001, 0.006));
  CHECK(budget.worthwhile(0.0001, 0.001));
  budget.remaining = 2u;
  CHECK_FALSE(budget.worthwhile(0.0001, 0.001));
  budget.remaining = 100u;
  budget.charge(0.005);
  CHECK_FALSE(budget.worthwhile(0.0001, 0.001));
  budget.remaining = 1000u;
  CHECK(budget.worthwhile(0.0001, 0.001));
  CHECK(budget.unseenGain(1.0) == 0.1);
  budget.unimprovedProposals = 3u;
  CHECK(budget.unseenGain(1.0) == 0.05);
  budget.remaining = 0u;
  budget.launched();
  CHECK(budget.remaining == 0u);
  CHECK_FALSE(budget.affordable(0.0));
  budget.remaining = std::numeric_limits<std::uint64_t>::max();
  CHECK(std::isfinite(budget.ceiling()));
  BenefitBudget zero(policy);
  zero.establishBaseline(0.0);
  CHECK_FALSE(zero.worthwhile(0.0, 0.0));
}

TEST_CASE("uncertainty targets close candidates and excludes clear losers",
          "[budget]") {
  using alpakaTune::detail::BenefitBudget;
  alpakaTune::detail::RuntimeStatistics incumbent;
  incumbent.median = 1.0;
  incumbent.acceptedSampleCount = 10u;
  auto tied = incumbent;
  auto loser = incumbent;
  loser.median = 10.0;
  CHECK(BenefitBudget::expectedGain(incumbent, tied) > 0.0);
  CHECK(BenefitBudget::expectedGain(incumbent, loser) == 0.0);
  tied.standardDeviation = 0.5;
  CHECK(BenefitBudget::expectedGain(incumbent, tied) > 0.01);
}

TEST_CASE("budget API rejects invalid policies and non-time objectives",
          "[budget]") {
  auto cfg = config();
  cfg.validate();
  cfg.budget->expectedLaunches = 0u;
  CHECK_THROWS(cfg.validate());
  cfg = config();
  cfg.budget->measurementCostHintSeconds = -1.0;
  CHECK_THROWS(cfg.validate());
  cfg.budget->measurementCostHintSeconds = 0.0;
  CHECK_NOTHROW(cfg.validate());
  auto dev = device();
  CHECK_THROWS(alpakaTune::makeTuner(
      cfg, alpakaTune::TunableBundle{choice(alpakaTune::RVals{0, 1})}, dev,
      alpakaTune::customMetric("energy")));
  CHECK_NOTHROW(alpakaTune::makeTuner(
      cfg, alpakaTune::TunableBundle{choice(alpakaTune::RVals{0, 1})}, dev,
      alpakaTune::elapsedTimeMetric("step_seconds",
                                    [](double a, double b) { return a + b; })));
}

TEST_CASE("budget YAML accepts optional hints without changing defaults",
          "[budget]") {
  auto path = std::filesystem::temp_directory_path() / "alpakaTune-budget.yaml";
  {
    std::ofstream file(path);
    file << "schema_version: 4\ntuning:\n  selection: fixed\n  "
            "runs_per_candidate: 1\n  budget:\n    expected_launches: 1000\n";
  }
  auto cfg = alpakaTune::TunerConfig::fromYaml(path);
  REQUIRE(cfg.budget);
  CHECK(cfg.budget->expectedLaunches == 1000u);
  CHECK(cfg.budget->maximumOverheadFraction == 0.05);
  CHECK_FALSE(cfg.budget->measurementCostHintSeconds);
  std::filesystem::remove(path);
}

TEST_CASE(
    "budgeted launches preserve output, geometry binding and invalidation",
    "[budget]") {
  auto const fixed = GENERATE(false, true);
  auto const queued = GENERATE(false, true);
  auto cfg = config();
  cfg.selection = fixed ? alpakaTune::SelectionPolicy::fixed
                        : alpakaTune::SelectionPolicy::adaptive;
  // A large optional hint guarantees economic stopping after bounded bootstrap.
  cfg.budget->measurementCostHintSeconds = 10000.0;
  if (queued)
    cfg.queue = alpakaTune::QueueConfig{.warmupRuns = 1u,
                                        .noiseCancellationWindow = 2u,
                                        .maxConsecutiveRuns = 2u};
  auto dev = device();
  auto queue =
      dev.makeQueue(alpaka::queueKind::nonBlocking, alpaka::timing::disabled);
  auto host = alpaka::onHost::allocHost<int>(Vec{2u});
  auto output = alpaka::onHost::allocLike(dev, host);
  host.getMdSpan()[0u] = 0;
  alpaka::onHost::memcpy(queue, output, host);
  auto frame =
      alpaka::onHost::FrameSpec{Vec{1u}, Vec{1u}, alpaka::exec::cpuSerial};
  auto bundle = alpaka::KernelBundle{Kernel{}, output.getMdSpan(),
                                     alpakaTune::markTunable(choice)};
  auto tuner = alpakaTune::makeTuner(
      cfg, alpakaTune::TunableBundle{choice(alpakaTune::RVals{0, 1})}, dev,
      alpakaTune::elapsedTimeMetric("step_seconds"),
      std::to_string(fixed) + std::to_string(queued));
  for (std::size_t launch = 0u; launch < 50u; ++launch) {
    auto observed = tuner.enqueueObserved(queue, frame, bundle);
    if (observed.purpose == alpakaTune::LaunchPurpose::experiment)
      tuner.provideMetric(1.0);
    if (launch > 10u) {
      CHECK(observed.purpose == alpakaTune::LaunchPurpose::production);
      CHECK_FALSE(observed.runtimeSeconds);
      CHECK(observed.recommendationSeconds == 0.0);
    }
  }
  REQUIRE(tuner.completed());
  REQUIRE(tuner.info().budget);
  CHECK(tuner.info().budget->measuredLaunches == 3u);
  CHECK(tuner.info().budget->remainingLaunches == 9950u);
  CHECK(tuner.info().executionCount == 50u);
  CHECK(tuner.history().size() == 50u);
  alpaka::onHost::memcpy(queue, host, output);
  alpaka::onHost::wait(queue);
  CHECK(host.getMdSpan()[0u] == 50);
  CHECK(host.getMdSpan()[1u] == 0);
  auto differentFrame =
      alpaka::onHost::FrameSpec{Vec{2u}, Vec{1u}, alpaka::exec::cpuSerial};
  CHECK_THROWS(tuner.enqueue(queue, differentFrame, bundle));
  tuner.setRemainingLaunches(0u);
  CHECK(tuner.enqueueObserved(queue, frame, bundle).purpose ==
        alpakaTune::LaunchPurpose::production);
  tuner.lastConfig().valid = false;
  tuner.lastConfig().valid = true;
  CHECK_NOTHROW(tuner.enqueue(queue, frame, bundle));
  tuner.lastConfig().valid = false;
  CHECK_THROWS(tuner.enqueue(queue, frame, bundle));
  CHECK_THROWS(tuner.lastConfig().valid = true);
  alpaka::onHost::wait(queue);
}

TEST_CASE("bootstrap respects the application horizon and launch guard",
          "[budget]") {
  auto const expected = GENERATE(1u, 2u, 3u);
  auto cfg = config();
  cfg.budget->expectedLaunches = expected;
  auto dev = device();
  auto queue =
      dev.makeQueue(alpaka::queueKind::nonBlocking, alpaka::timing::disabled);
  auto host = alpaka::onHost::allocHost<int>(Vec{2u});
  auto output = alpaka::onHost::allocLike(dev, host);
  auto frame =
      alpaka::onHost::FrameSpec{Vec{1u}, Vec{1u}, alpaka::exec::cpuSerial};
  // nvcc defect (CUDA 12.5–13.0): rejects a CVals binding nested directly
  // in TunableBundle CTAD. Bind it separately; see README's known limitations.
  auto const tunable = choice(alpakaTune::CVals<0, 1>{});
  auto tuner = alpakaTune::makeTuner(
      cfg, alpakaTune::TunableBundle{tunable}, dev,
      alpakaTune::elapsedTimeMetric("runtime"), std::to_string(expected));
  auto bundle = alpaka::KernelBundle{Kernel{}, output.getMdSpan(),
                                     alpakaTune::markTunable(choice)};
  for (unsigned i = 0u; i < 8u; ++i) {
    auto observed = tuner.enqueueObserved(queue, frame, bundle);
    if (observed.purpose == alpakaTune::LaunchPurpose::experiment)
      tuner.provideMetric(1.0);
  }
  REQUIRE(tuner.info().budget);
  CHECK(tuner.info().budget->remainingLaunches == 0u);
  CHECK(tuner.info().budget->measuredLaunches == expected);
  CHECK(tuner.completed());
  alpaka::onHost::wait(queue);
}

TEST_CASE("sparse checks revalidate known alternatives after a phase change",
          "[budget]") {
  auto cfg = config();
  cfg.budget->expectedLaunches = 100000u;
  cfg.budget->checkInterval = 4u;
  cfg.maximumExecutions = 24u;
  cfg.historyWindowSize = 10u;
  auto dev = device();
  auto queue =
      dev.makeQueue(alpaka::queueKind::nonBlocking, alpaka::timing::disabled);
  auto host = alpaka::onHost::allocHost<int>(Vec{2u});
  auto output = alpaka::onHost::allocLike(dev, host);
  host.getMdSpan()[0u] = 0;
  alpaka::onHost::memcpy(queue, output, host);
  auto frame =
      alpaka::onHost::FrameSpec{Vec{1u}, Vec{1u}, alpaka::exec::cpuSerial};
  auto tuner = alpakaTune::makeTuner(
      cfg, alpakaTune::TunableBundle{choice(alpakaTune::RVals{0, 1})}, dev,
      alpakaTune::elapsedTimeMetric("phase_seconds"));
  auto bundle = alpaka::KernelBundle{Kernel{}, output.getMdSpan(),
                                     alpakaTune::markTunable(choice)};
  for (unsigned i = 0u; i < 40u && !tuner.completed(); ++i) {
    auto observed = tuner.enqueueObserved(queue, frame, bundle);
    if (observed.purpose == alpakaTune::LaunchPurpose::experiment)
      tuner.provideMetric(observed.candidateIndex == 0u ? 1.0 : 2.0);
  }
  REQUIRE(tuner.completed());
  REQUIRE(tuner.info().measuredCandidateCount == 2u);
  REQUIRE(tuner.bestCandidateIndex() == 0u);
  auto unseen = tuner.info().unseenAcceptedCount;
  auto retired = tuner.info().retiredConfigurationCount;
  auto measurements = tuner.info().budget->measuredLaunches;
  auto checked = 0u;
  for (unsigned i = 0u; i < 36u; ++i) {
    auto observed = tuner.enqueueObserved(queue, frame, bundle);
    if (observed.purpose == alpakaTune::LaunchPurpose::healthCheck ||
        observed.purpose == alpakaTune::LaunchPurpose::experiment) {
      tuner.provideMetric(observed.candidateIndex == 0u ? 4.0 : 0.5);
      ++checked;
    }
  }
  CHECK(checked < 20u);
  CHECK(tuner.info().budget->measuredLaunches > measurements);
  CHECK(tuner.bestCandidateIndex() == 1u);
  CHECK(tuner.info().unseenAcceptedCount == unseen);
  CHECK(tuner.info().retiredConfigurationCount == retired);
  CHECK(tuner.completed());
  alpaka::onHost::wait(queue);
}

TEST_CASE(
    "missing elapsed metrics reject candidates and preserve a valid fallback",
    "[budget]") {
  auto cfg = config();
  auto dev = device();
  auto queue =
      dev.makeQueue(alpaka::queueKind::nonBlocking, alpaka::timing::disabled);
  auto host = alpaka::onHost::allocHost<int>(Vec{2u});
  auto output = alpaka::onHost::allocLike(dev, host);
  host.getMdSpan()[0u] = 0;
  alpaka::onHost::memcpy(queue, output, host);
  auto frame =
      alpaka::onHost::FrameSpec{Vec{1u}, Vec{1u}, alpaka::exec::cpuSerial};
  auto tuner = alpakaTune::makeTuner(
      cfg, alpakaTune::TunableBundle{choice(alpakaTune::RVals{0, 1, 2})}, dev,
      alpakaTune::elapsedTimeMetric("missing_seconds"));
  auto bundle = alpaka::KernelBundle{Kernel{}, output.getMdSpan(),
                                     alpakaTune::markTunable(choice)};
  for (unsigned i = 0u; i < 20u; ++i) {
    auto observed = tuner.enqueueObserved(queue, frame, bundle);
    if (observed.purpose == alpakaTune::LaunchPurpose::experiment &&
        observed.candidateIndex != 1u)
      tuner.provideMetric(1.0);
  }
  CHECK(tuner.info().missingMetricCandidateCount == 1u);
  CHECK(tuner.lastCandidateIndex() != 1u);
  alpaka::onHost::wait(queue);
}

TEST_CASE("budgeted automatic catalogs retain stable candidate IDs",
          "[budget]") {
  auto cfg = config();
  cfg.space.initialCandidates = 2u;
  cfg.space.maximumCandidates = 4u;
  cfg.maximumExecutions = 30u;
  auto dev = device();
  auto queue =
      dev.makeQueue(alpaka::queueKind::nonBlocking, alpaka::timing::disabled);
  auto host = alpaka::onHost::allocHost<int>(Vec{2u});
  auto output = alpaka::onHost::allocLike(dev, host);
  host.getMdSpan()[0u] = 0;
  alpaka::onHost::memcpy(queue, output, host);
  auto frame =
      alpaka::onHost::FrameSpec{Vec{1u}, Vec{1u}, alpaka::exec::cpuSerial};
  auto tuner = alpakaTune::makeTuner(
      cfg,
      alpakaTune::TunableBundle{choice(
          alpakaTune::autoCandidates(alpakaTune::domain::interval(0, 3)))},
      dev, alpakaTune::elapsedTimeMetric("catalog_seconds"));
  auto bundle = alpaka::KernelBundle{Kernel{}, output.getMdSpan(),
                                     alpakaTune::markTunable(choice)};
  for (unsigned i = 0u; i < 50u; ++i) {
    auto observed = tuner.enqueueObserved(queue, frame, bundle);
    CHECK(observed.candidateIndex < tuner.info().candidateCount);
    if (observed.purpose == alpakaTune::LaunchPurpose::experiment)
      tuner.provideMetric(1.0 + observed.candidateIndex);
  }
  CHECK(tuner.completed());
  CHECK(tuner.info().candidateCount <= 4u);
  alpaka::onHost::wait(queue);
}

#if ALPAKA_TUNE_HAS_JSON
TEST_CASE("budgeted persistence supports both offline policies and fresh cost "
          "accounting",
          "[budget]") {
  auto const complete = GENERATE(false, true);
  auto const fixed = GENERATE(false, true);
  auto path = std::filesystem::temp_directory_path() /
              ("alpakaTune-budget-" + std::to_string(complete) +
               std::to_string(fixed) + ".json");
  std::filesystem::remove(path);
  auto cfg = config();
  cfg.selection = alpakaTune::SelectionPolicy::fixed;
  cfg.maximumExecutions = 4u;
  if (complete)
    cfg.completeHistory = {.file = path, .read = true, .write = true};
  else
    cfg.history = {.file = path, .read = true, .write = true};
  auto dev = device();
  auto queue =
      dev.makeQueue(alpaka::queueKind::nonBlocking, alpaka::timing::disabled);
  auto host = alpaka::onHost::allocHost<int>(Vec{2u});
  auto output = alpaka::onHost::allocLike(dev, host);
  host.getMdSpan()[0u] = 0;
  alpaka::onHost::memcpy(queue, output, host);
  auto frame =
      alpaka::onHost::FrameSpec{Vec{1u}, Vec{1u}, alpaka::exec::cpuSerial};
  auto tunables = alpakaTune::TunableBundle{choice(alpakaTune::RVals{0, 1})};
  auto bundle = alpaka::KernelBundle{Kernel{}, output.getMdSpan(),
                                     alpakaTune::markTunable(choice)};
  auto identity = std::to_string(complete) + std::to_string(fixed);
  {
    auto trainer =
        alpakaTune::makeTuner(cfg, tunables, dev, identity,
                              alpakaTune::elapsedTimeMetric("saved_seconds"));
    for (unsigned i = 0u; i < 6u; ++i) {
      auto observed = trainer.enqueueObserved(queue, frame, bundle);
      if (observed.purpose == alpakaTune::LaunchPurpose::experiment)
        trainer.provideMetric(1.0);
    }
    REQUIRE(trainer.completed());
  }
  cfg.exploration = alpakaTune::ExplorationPolicy::offline;
  cfg.selection = fixed ? alpakaTune::SelectionPolicy::fixed
                        : alpakaTune::SelectionPolicy::adaptive;
  cfg.maximumExecutions.reset();
  cfg.history.write = false;
  cfg.completeHistory.write = false;
  auto replay =
      alpakaTune::makeTuner(cfg, tunables, dev, identity,
                            alpakaTune::elapsedTimeMetric("saved_seconds"));
  for (unsigned i = 0u; i < 20u; ++i) {
    auto observed = replay.enqueueObserved(queue, frame, bundle);
    CHECK(observed.purpose == alpakaTune::LaunchPurpose::production);
    CHECK(replay.loadedFromCache());
  }
  CHECK(replay.info().budget->measuredLaunches == 0u);
  CHECK(replay.info().budget->remainingLaunches == 9980u);
  // Offline fixed diagnostics may retain collected admission counts; no
  // exploration takes place and no new measurements are collected.
  CHECK(replay.info().measuredCandidateCount > 0u);
  CHECK(replay.completed());
  alpaka::onHost::wait(queue);
  std::filesystem::remove(path);
}
#endif

namespace {
template <typename Queue> struct CountedQueue {
  Queue queue;
  mutable std::size_t events{};
  mutable std::size_t kernels{};
  auto getDevice() const { return queue.getDevice(); }
  auto getTiming() const { return queue.getTiming(); }
  auto getQueueKind() const { return queue.getQueueKind(); }
  void enqueue(auto const &event) const {
    ++events;
    queue.enqueue(event);
  }
  void enqueue(auto const &spec, auto const &bundle) const {
    ++kernels;
    queue.enqueue(spec, bundle);
  }
};
} // namespace

TEST_CASE("ordinary budget launches enqueue one kernel and no timer events",
          "[budget]") {
  auto cfg = config();
  cfg.budget->expectedLaunches = 1u;
  auto dev = device();
  auto rawQueue =
      dev.makeQueue(alpaka::queueKind::nonBlocking, alpaka::timing::enabled);
  CountedQueue counted{rawQueue};
  auto host = alpaka::onHost::allocHost<int>(Vec{2u});
  auto output = alpaka::onHost::allocLike(dev, host);
  host.getMdSpan()[0u] = 0;
  alpaka::onHost::memcpy(rawQueue, output, host);
  auto frame =
      alpaka::onHost::FrameSpec{Vec{1u}, Vec{1u}, alpaka::exec::cpuSerial};
  auto tuner = alpakaTune::makeTuner(
      cfg, alpakaTune::TunableBundle{choice(alpakaTune::RVals{0, 1})}, dev,
      "counted-events");
  auto bundle = alpaka::KernelBundle{Kernel{}, output.getMdSpan(),
                                     alpakaTune::markTunable(choice)};
  tuner.enqueue(counted, frame, bundle);
  CHECK(counted.events == 2u);
  for (unsigned i = 0u; i < 50u; ++i)
    tuner.enqueue(counted, frame, bundle);
  CHECK(counted.events == 2u);
  CHECK(counted.kernels == 51u);
  CHECK(tuner.info().budget->measuredLaunches == 1u);
  alpaka::onHost::memcpy(rawQueue, host, output);
  alpaka::onHost::wait(rawQueue);
  CHECK(host.getMdSpan()[0u] == 51);
}
