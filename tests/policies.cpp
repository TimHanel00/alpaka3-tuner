// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>
#include <alpakaTune/alpakaTune.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace {
inline constexpr auto choice = ALPAKA_TUNE_TUNABLE("choice");
struct WriteChoice {
  ALPAKA_FN_ACC void operator()(auto const &,
                                alpaka::concepts::IMdSpan auto output,
                                int value) const {
    output[0u] = value;
  }
};
} // namespace

TEST_CASE("exploration completion and selection are independent",
          "[policies]") {
  auto const exploration = GENERATE(alpakaTune::ExplorationPolicy::online,
                                    alpakaTune::ExplorationPolicy::offline);
  auto const selection = GENERATE(alpakaTune::SelectionPolicy::fixed,
                                  alpakaTune::SelectionPolicy::adaptive);
  auto const complete = GENERATE(false, true);
  auto const knownCount = GENERATE(1u, 3u);
  CAPTURE(exploration, selection, complete, knownCount);
#if !ALPAKA_TUNE_HAS_JSON
  if (exploration == alpakaTune::ExplorationPolicy::offline)
    SKIP("Offline reuse requires persistence support.");
#endif
  using Index = alpaka::Vec<std::size_t, 1u>;
  auto selector = alpaka::onHost::makeDeviceSelector(
      alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu});
  if (!selector.isAvailable())
    SKIP("The host backend is unavailable.");
  auto device = selector.makeDevice(0u);
  auto queue = alpakaTune::makeQueue(device, alpaka::queueKind::nonBlocking,
                                     alpakaTune::timing::enabled);
  auto host = alpaka::onHost::allocHost<int>(Index{1u});
  auto output = alpaka::onHost::allocLike(device, host);
  auto const frame =
      alpaka::onHost::FrameSpec{Index{1u}, Index{1u}, alpaka::exec::cpuSerial};
  auto const tunables =
      alpakaTune::TunableBundle{choice(alpakaTune::RVals{0, 1, 2, 3})};
  auto const bundle = alpaka::KernelBundle{WriteChoice{}, output.getMdSpan(),
                                           alpakaTune::markTunable(choice)};
  auto config = alpakaTune::TunerConfig{};
  config.selection = selection;
  config.maximumExecutions = knownCount;
  config.historyWindowSize = 1u;
  config.adaptiveProbeInterval = 10u;
  auto const identity =
      std::string{"policy-matrix-"} +
      std::string{alpakaTune::explorationPolicyName(exploration)} + "-" +
      std::string{alpakaTune::selectionPolicyName(selection)} +
      (complete ? "-complete" : "-compact") + std::to_string(knownCount);
  auto const path =
      std::filesystem::temp_directory_path() / (identity + ".json");
  std::filesystem::remove(path);
  if (exploration == alpakaTune::ExplorationPolicy::offline) {
    if (complete)
      config.completeHistory.file = path;
    else
      config.history.file = path;
    {
      auto training = config;
      training.selection = alpakaTune::SelectionPolicy::fixed;
      auto trainer =
          alpakaTune::makeTuner(training, tunables, device, identity,
                                alpakaTune::customMetric("objective"));
      while (!trainer.completed()) {
        trainer.enqueue(queue, frame, bundle);
        trainer.provideMetric(trainer.lastCandidateIndex() == 0u ? 1.0 : 10.0);
      }
      REQUIRE(trainer.info().measuredCandidateCount == knownCount);
    }
    config.exploration = exploration;
    config.maximumExecutions.reset();
    // Queue warmups must not consume known-only measurements.
    config.queue =
        alpakaTune::QueueConfig{.warmupRuns = 2u, .maxConsecutiveRuns = 3u};
  }
  auto tuner = alpakaTune::makeTuner(config, tunables, device, identity,
                                     alpakaTune::customMetric("objective"));
  if (exploration == alpakaTune::ExplorationPolicy::online) {
    while (!tuner.completed()) {
      tuner.enqueue(queue, frame, bundle);
      tuner.provideMetric(tuner.lastCandidateIndex() == 0u ? 1.0 : 10.0);
    }
  } else {
    tuner.enqueue(queue, frame, bundle);
    tuner.provideMetric(1.0);
  }
  REQUIRE(tuner.completed());
  REQUIRE(tuner.info().explorationComplete);
  REQUIRE(tuner.info().measuredCandidateCount == knownCount);
  REQUIRE(tuner.info().selectionLocked ==
          (selection == alpakaTune::SelectionPolicy::fixed));
  auto const retired = tuner.info().retiredConfigurationCount;
  auto const unseen = tuner.info().unseenAcceptedCount;
  auto const size = tuner.info().candidateCount;
  auto const priorReuses =
      exploration == alpakaTune::ExplorationPolicy::offline &&
              selection == alpakaTune::SelectionPolicy::adaptive
          ? 1u
          : 0u;
  for (std::size_t launch = priorReuses; launch < 30u; ++launch) {
    auto const observation = tuner.enqueueObserved(queue, frame, bundle);
    auto const candidate = observation.candidateIndex;
    REQUIRE(candidate <
            knownCount); // The fourth candidate has never been measured.
    if (selection == alpakaTune::SelectionPolicy::fixed || knownCount == 1u)
      CHECK(candidate == 0u);
    else if (launch < 9u)
      CHECK(candidate == 0u);
    else if (launch == 9u)
      CHECK(candidate == 1u);
    else if (launch == 19u)
      CHECK(candidate == 2u);
    else if (launch == 29u)
      CHECK(candidate == 0u);
    tuner.provideMetric(launch < 9u ? (candidate == 0u ? 1.0 : 10.0)
                                    : (candidate == 0u   ? 20.0
                                       : candidate == 1u ? 0.5
                                                         : 15.0));
    alpaka::onHost::memcpy(queue, host, output);
    alpaka::onHost::wait(queue);
    CHECK(host.getMdSpan()[0u] == static_cast<int>(candidate));
    CHECK(tuner.lastConfig().measured ==
          (selection == alpakaTune::SelectionPolicy::adaptive));
    CHECK(tuner.candidateMetricSamples(candidate).size() == 1u);
  }
  CHECK(tuner.bestCandidateIndex() ==
        (selection == alpakaTune::SelectionPolicy::adaptive && knownCount > 1u
             ? 1u
             : 0u));
  CHECK(tuner.info().candidateCount == size);
  CHECK(tuner.info().unseenAcceptedCount == unseen);
  CHECK(tuner.info().retiredConfigurationCount == retired);
  CHECK(tuner.completed());
  if (selection == alpakaTune::SelectionPolicy::adaptive && knownCount > 1u) {
    for (auto const &execution : tuner.history())
      if (execution.candidateIndex == 1u)
        execution.valid = false;
    tuner.enqueue(queue, frame, bundle);
    CHECK(tuner.lastCandidateIndex() != 1u);
    tuner.provideMetric(5.0);
    tuner.enqueue(queue, frame, bundle);
    auto const missingMetricCandidate = tuner.lastCandidateIndex();
    tuner.enqueue(queue, frame, bundle);
    CHECK(tuner.lastCandidateIndex() != missingMetricCandidate);
    tuner.provideMetric(5.0);
    CHECK(tuner.info().missingMetricCandidateCount == 1u);
    // Removing the remaining known candidates must never start a new search.
    for (auto const &execution : tuner.history())
      execution.valid = false;
    CHECK_THROWS(tuner.enqueue(queue, frame, bundle));
    CHECK(tuner.completionReason() ==
          alpakaTune::TunerCompletionReason::noValidConfiguration);
  }
  if (selection == alpakaTune::SelectionPolicy::adaptive && knownCount == 1u) {
    tuner.lastConfig().valid = false;
    CHECK_THROWS(tuner.enqueue(queue, frame, bundle));
    CHECK(tuner.completionReason() ==
          alpakaTune::TunerCompletionReason::noValidConfiguration);
  }
  alpaka::onHost::wait(queue);
  std::filesystem::remove(path);
}

TEST_CASE("policy YAML validates independent axes and migration",
          "[policies]") {
  auto const path =
      std::filesystem::temp_directory_path() / "alpakaTune-policy-yaml.yaml";
  auto load = [&](std::string const &yaml) {
    {
      std::ofstream output{path};
      output << yaml;
    }
    return alpakaTune::TunerConfig::fromYaml(path);
  };
  for (auto const exploration : {"online", "offline"}) {
    for (auto const selection : {"fixed", "adaptive"}) {
      auto yaml = std::string{"schema_version: 4\ntuning:\n  "
                              "runs_per_candidate: 1\n  exploration: "} +
                  exploration + "\n  selection: " + selection +
                  "\n  adaptive_probe_interval: 7\n";
      if (std::string{exploration} == "online")
        yaml += "  maximum_executions: 3\n";
      auto const config = load(yaml);
      CHECK(alpakaTune::explorationPolicyName(config.exploration) ==
            exploration);
      CHECK(alpakaTune::selectionPolicyName(config.selection) == selection);
      CHECK(config.adaptiveProbeInterval == 7u);
    }
  }
  for (auto const version : {1, 2, 3})
    CHECK_THROWS_WITH(load("schema_version: " + std::to_string(version) +
                           "\ntuning:\n  mode: online_fixed\n"),
                      "Unsupported alpakaTune YAML schema_version; migrate to "
                      "schema_version: 4 with tuning.exploration and "
                      "tuning.selection instead of tuning.mode.");
  CHECK_THROWS(load("schema_version: 4\ntuning:\n  mode: online_fixed\n  "
                    "runs_per_candidate: 1\n"));
  for (auto const field : {"exploration: invalid", "selection: invalid",
                           "adaptive_probe_interval: 0", "selection: fixed",
                           "replay_fast_path: true"})
    CHECK_THROWS(load(
        std::string{"schema_version: 4\ntuning:\n  runs_per_candidate: 1\n  "} +
        field + "\n"));
  std::filesystem::remove(path);
}

#if ALPAKA_TUNE_HAS_JSON
TEST_CASE("timed known configurations survive a process boundary",
          "[policy-process]") {
  auto const *phase = std::getenv("ALPAKA_TUNE_POLICY_PROCESS");
  if (!phase)
    SKIP("Run through the policy process CTest fixture.");
  auto const stage = std::string{phase} == "stage";
  auto const compact = GENERATE(false, true);
  auto const selection = GENERATE(alpakaTune::SelectionPolicy::fixed,
                                  alpakaTune::SelectionPolicy::adaptive);
  using Index = alpaka::Vec<std::size_t, 1u>;
  auto device = alpaka::onHost::makeDeviceSelector(
                    alpaka::onHost::DeviceSpec{alpaka::api::host,
                                               alpaka::deviceKind::cpu})
                    .makeDevice(0u);
  auto queue = alpakaTune::makeQueue(device, alpaka::queueKind::nonBlocking,
                                     alpakaTune::timing::enabled);
  auto host = alpaka::onHost::allocHost<int>(Index{1u});
  auto output = alpaka::onHost::allocLike(device, host);
  auto const frame =
      alpaka::onHost::FrameSpec{Index{1u}, Index{1u}, alpaka::exec::cpuSerial};
  auto const tunables =
      alpakaTune::TunableBundle{choice(alpakaTune::RVals{0, 1, 2, 3})};
  auto const bundle = alpaka::KernelBundle{WriteChoice{}, output.getMdSpan(),
                                           alpakaTune::markTunable(choice)};
  auto const path = std::filesystem::path{
      compact ? "policy-process-compact.json" : "policy-process-complete.json"};
  if (stage)
    std::filesystem::remove(path);
  auto readBytes = [&] {
    std::ifstream input{path};
    return std::string{std::istreambuf_iterator<char>{input},
                       std::istreambuf_iterator<char>{}};
  };
  auto const original = stage ? std::string{} : readBytes();
  {
    auto config = alpakaTune::TunerConfig{};
    config.exploration = stage ? alpakaTune::ExplorationPolicy::online
                               : alpakaTune::ExplorationPolicy::offline;
    config.selection = stage ? alpakaTune::SelectionPolicy::fixed : selection;
    config.historyWindowSize = 3u;
    if (stage)
      config.maximumExecutions = 3u;
    if (compact)
      config.history = {.file = path, .read = !stage, .write = stage};
    else
      config.completeHistory = {.file = path, .read = !stage, .write = stage};
    auto tuner = alpakaTune::makeTuner(config, tunables, device,
                                       "policy-process-timing");
    auto const launches = stage ? 3u : 30u;
    for (std::size_t launch = 0u; launch < launches; ++launch) {
      auto const observation = tuner.enqueueObserved(queue, frame, bundle);
      REQUIRE(observation.candidateIndex < 3u);
      CHECK(observation.measured ==
            (stage || selection == alpakaTune::SelectionPolicy::adaptive));
      CHECK(observation.runtimeSeconds.has_value() == observation.measured);
      if (!stage)
        CHECK(observation.recommendationSeconds == 0.0);
      alpaka::onHost::memcpy(queue, host, output);
      alpaka::onHost::wait(queue);
      CHECK(host.getMdSpan()[0u] ==
            static_cast<int>(observation.candidateIndex));
    }
    CHECK(tuner.completed());
    CHECK(tuner.info().measuredCandidateCount == 3u);
    if (!stage) {
      CHECK(tuner.loadedFromCache());
      if (selection == alpakaTune::SelectionPolicy::adaptive) {
        std::size_t retained{};
        for (std::size_t candidate = 0u; candidate < 3u; ++candidate) {
          CHECK(tuner.candidateRuntimeSamples(candidate).size() <= 3u);
          retained += tuner.candidateRuntimeSamples(candidate).size();
        }
        CHECK(retained > 3u);
      }
    }
  }
  if (!stage)
    CHECK(readBytes() == original);
}
#endif

#if ALPAKA_TUNE_HAS_JSON
TEST_CASE("offline adaptation preserves saved learned state without inference",
          "[policy-process]") {
  auto const *phase = std::getenv("ALPAKA_TUNE_POLICY_PROCESS");
  if (!phase || std::string{phase} != "replay")
    SKIP("Requires the process fixture's collected timing history.");
  auto const compact = GENERATE(false, true);
  auto const source = std::filesystem::path{
      compact ? "policy-process-compact.json" : "policy-process-complete.json"};
  auto const target = std::filesystem::path{
      compact ? "policy-adapter-compact.json" : "policy-adapter-complete.json"};
  auto const model = std::filesystem::path{"policy-unused-model.atml"};
  {
    std::ofstream output{model};
    output << "Known reuse must not run model inference.";
  }
  auto const digest = alpakaTune::detail::fileFingerprint(model);
  auto saved = nlohmann::json{};
  {
    std::ifstream input{source};
    input >> saved;
  }
  auto adapter = nlohmann::json{{"state_version", 1},
                                {"coefficients", {1.0, 2.0}},
                                {"observations_since_update", 2},
                                {"update_count", 4},
                                {"observations", nlohmann::json::array()}};
  for (auto &[key, context] : saved["contexts"].items()) {
    static_cast<void>(key);
    // Legacy policy metadata must not invalidate compatible measurements.
    if (!compact) {
      context["metadata"].erase("exploration");
      context["metadata"].erase("selection");
      context["metadata"]["mode"] = "online_fixed";
      context.erase("exploration_complete");
    }
    if (compact) {
      context["adapter"] = adapter;
      context["adapter"]["model_digest"] = digest;
    } else
      context["learning"] = {{"status", "active"},
                             {"model_digest", digest},
                             {"residual_adapter", adapter}};
  }
  {
    std::ofstream output{target};
    output << saved;
  }
  using Index = alpaka::Vec<std::size_t, 1u>;
  auto device = alpaka::onHost::makeDeviceSelector(
                    alpaka::onHost::DeviceSpec{alpaka::api::host,
                                               alpaka::deviceKind::cpu})
                    .makeDevice(0u);
  auto queue = alpakaTune::makeQueue(device, alpaka::queueKind::nonBlocking,
                                     alpakaTune::timing::enabled);
  auto host = alpaka::onHost::allocHost<int>(Index{1u});
  auto output = alpaka::onHost::allocLike(device, host);
  auto const frame =
      alpaka::onHost::FrameSpec{Index{1u}, Index{1u}, alpaka::exec::cpuSerial};
  auto const tunables =
      alpakaTune::TunableBundle{choice(alpakaTune::RVals{0, 1, 2, 3})};
  {
    auto config = alpakaTune::TunerConfig{};
    config.exploration = alpakaTune::ExplorationPolicy::offline;
    config.strategy = alpakaTune::StrategyKind::learnedHybrid;
    config.learnedModelFile = model;
    if (compact)
      config.history.file = target;
    else
      config.completeHistory.file = target;
    auto tuner = alpakaTune::makeTuner(config, tunables, device,
                                       "policy-process-timing");
    auto observation = tuner.enqueueObserved(
        queue, frame,
        alpaka::KernelBundle{WriteChoice{}, output.getMdSpan(),
                             alpakaTune::markTunable(choice)});
    CHECK(observation.measured);
    CHECK_FALSE(observation.learnedStatus.has_value());
    CHECK(tuner.loadedFromCache());
    CHECK(tuner.completed());
  }
  auto restored = nlohmann::json{};
  {
    std::ifstream input{target};
    input >> restored;
  }
  for (auto const &[key, context] : restored["contexts"].items()) {
    static_cast<void>(key);
    auto actual = compact ? context.at("adapter")
                          : context.at("learning").at("residual_adapter");
    if (compact)
      actual.erase("model_digest");
    CHECK(actual == adapter);
  }
  std::filesystem::remove(target);
  std::filesystem::remove(model);
}
#endif
