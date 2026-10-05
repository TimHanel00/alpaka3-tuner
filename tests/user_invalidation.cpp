// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>
#include <alpakaTune/alpakaTune.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

#if ALPAKA_TUNE_HAS_JSON
#include <nlohmann/json.hpp>
#endif

namespace {

inline constexpr auto qualityLevel = ALPAKA_TUNE_TUNABLE("qualityLevel");

struct WriteKernel {
  ALPAKA_FN_ACC void operator()(auto const &acc,
                                alpaka::concepts::IMdSpan auto output,
                                int value) const {
    static_cast<void>(acc);
    output[0u] = value;
  }
};

[[nodiscard]] auto fixedConfig(bool queued) -> alpakaTune::TunerConfig {
  auto config = alpakaTune::TunerConfig{};
  config.exploration = alpakaTune::ExplorationPolicy::online;
  config.selection = alpakaTune::SelectionPolicy::fixed;
  config.strategy = alpakaTune::StrategyKind::exhaustive;
  config.queue =
      queued
          ? std::optional{alpakaTune::QueueConfig{.disable = false,
                                                  .warmupRuns = 0u,
                                                  .noiseCancellationWindow = 3u,
                                                  .maxConsecutiveRuns = 1u}}
          : std::nullopt;
  config.runsPerCandidate = 2u;
  config.minimumRunsPerCandidate = 2u;
  config.mannWhitneyEarlyStop = false;
  config.maximumExecutions = 100u;
  config.maximumRetiredConfigurations.reset();
  config.history.file.reset();
  config.history.read = false;
  config.history.write = false;
  config.completeHistory.file.reset();
  config.completeHistory.read = false;
  config.completeHistory.write = false;
  return config;
}

template <typename Device> struct Fixture {
  using Index = alpaka::Vec<std::size_t, 1u>;
  using Queue = ALPAKA_TYPEOF(alpakaTune::makeQueue(
      std::declval<Device &>(), alpaka::queueKind::nonBlocking,
      alpakaTune::timing::enabled));
  using HostBuffer = ALPAKA_TYPEOF(alpaka::onHost::allocHost<int>(Index{1u}));
  using DeviceBuffer = ALPAKA_TYPEOF(alpaka::onHost::allocLike(
      std::declval<Device &>(), std::declval<HostBuffer &>()));

  Device device;
  Queue queue;
  HostBuffer host;
  DeviceBuffer output;
  ALPAKA_TYPEOF(alpaka::onHost::FrameSpec{Index{1u}, Index{1u},
                                          alpaka::exec::cpuSerial})
  frameSpec;

  explicit Fixture(Device selected)
      : device(std::move(selected)),
        queue(alpakaTune::makeQueue(device, alpaka::queueKind::nonBlocking,
                                    alpakaTune::timing::enabled)),
        host(alpaka::onHost::allocHost<int>(Index{1u})),
        output(alpaka::onHost::allocLike(device, host)),
        frameSpec{Index{1u}, Index{1u}, alpaka::exec::cpuSerial} {}

  ~Fixture() {
    // Timing-disabled launches and terminal replays may still be queued.
    // Drain them while every span target is alive; members are destroyed
    // before the queue member would otherwise synchronize during teardown.
    alpaka::onHost::wait(queue);
  }
};

[[nodiscard]] auto hostDevice() {
  auto selector = alpaka::onHost::makeDeviceSelector(
      alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu});
  using Device = ALPAKA_TYPEOF(selector.makeDevice(0u));
  if (!selector.isAvailable())
    return std::optional<Device>{};
  return std::optional{selector.makeDevice(0u)};
}

} // namespace

TEST_CASE("post-evaluation invalidation preserves actual execution order",
          "[validity]") {
  auto selected = hostDevice();
  if (!selected)
    SKIP("The host backend is unavailable.");
  auto fixture = Fixture{*selected};
  auto const tunables =
      alpakaTune::TunableBundle{qualityLevel(alpakaTune::RVals{1, 2, 3})};
  auto tuner = alpakaTune::makeTuner(fixedConfig(false), tunables,
                                     fixture.device, "validity-direct");
  auto const bundle = alpaka::KernelBundle{
      WriteKernel{}, fixture.output.getMdSpan(), qualityLevel};

  CHECK(tuner.history().empty());
  CHECK(tuner.lastCandidateIndex() == std::numeric_limits<std::size_t>::max());
  CHECK_THROWS_AS(tuner.lastConfig(), std::logic_error);

  tuner.enqueue(fixture.queue, fixture.frameSpec, bundle);
  CHECK(tuner.history().size() == 1u);
  CHECK(tuner.lastConfig().candidateIndex == 0u);
  CHECK(tuner.lastConfig().configuration ==
        alpakaTune::ParameterConfiguration{0.0f});
  tuner.lastConfig().valid = false;

  tuner.enqueue(fixture.queue, fixture.frameSpec, bundle);
  CHECK(tuner.lastConfig().candidateIndex == 1u);
  tuner.lastConfig().valid = false;

  while (!tuner.completed())
    tuner.enqueue(fixture.queue, fixture.frameSpec, bundle);

  REQUIRE(tuner.history().size() == 4u);
  CHECK(tuner.history()[0u].candidateIndex == 0u);
  CHECK(tuner.history()[1u].candidateIndex == 1u);
  CHECK(tuner.history()[2u].candidateIndex == 2u);
  CHECK(tuner.history()[3u].candidateIndex == 2u);
  CHECK_FALSE(static_cast<bool>(tuner.history()[0u].valid));
  CHECK_FALSE(static_cast<bool>(tuner.history()[1u].valid));
  CHECK(static_cast<bool>(tuner.history()[2u].valid));
  CHECK(tuner.lastCandidateIndex() == tuner.history().back().candidateIndex);
  CHECK(tuner.bestCandidateIndex() == 2u);
  CHECK(tuner.info().userInvalidatedCandidateCount == 2u);
  CHECK(tuner.info().userInvalidatedRejectedCount >= 2u);
  CHECK(tuner.info().restrictionRejectedCount == 0u);
  CHECK(tuner.info().measuredCandidateCount == 1u);
  CHECK(tuner.candidateRuntimeSamples(0u).size() == 1u);
  CHECK_THROWS_AS(tuner.history().front().valid = true, std::logic_error);
}

TEST_CASE("queued invalid candidates are retired before the next launch",
          "[validity][scheduler]") {
  auto selected = hostDevice();
  if (!selected)
    SKIP("The host backend is unavailable.");
  auto fixture = Fixture{*selected};
  auto const tunables =
      alpakaTune::TunableBundle{qualityLevel(alpakaTune::RVals{1, 2, 3})};
  auto tuner = alpakaTune::makeTuner(fixedConfig(true), tunables,
                                     fixture.device, "validity-queue");
  auto const bundle = alpaka::KernelBundle{
      WriteKernel{}, fixture.output.getMdSpan(), qualityLevel};
  auto launches = std::array<std::size_t, 3u>{};

  while (!tuner.completed()) {
    tuner.enqueue(fixture.queue, fixture.frameSpec, bundle);
    auto &executed = tuner.lastConfig();
    ++launches.at(executed.candidateIndex);
    if (executed.candidateIndex < 2u)
      executed.valid = false;
    REQUIRE(tuner.history().size() < 10u);
  }

  CHECK(launches == std::array<std::size_t, 3u>{1u, 1u, 2u});
  CHECK(tuner.bestCandidateIndex() == 2u);
}

TEST_CASE("invalidating the only candidate terminates without a winner",
          "[validity]") {
  auto selected = hostDevice();
  if (!selected)
    SKIP("The host backend is unavailable.");
  auto fixture = Fixture{*selected};
  auto const tunables =
      alpakaTune::TunableBundle{qualityLevel(alpakaTune::RVals{1})};
  auto tuner = alpakaTune::makeTuner(fixedConfig(false), tunables,
                                     fixture.device, "validity-no-winner");
  auto const bundle = alpaka::KernelBundle{
      WriteKernel{}, fixture.output.getMdSpan(), qualityLevel};

  tuner.enqueue(fixture.queue, fixture.frameSpec, bundle);
  tuner.lastConfig().valid = false;
  CHECK_THROWS(tuner.enqueue(fixture.queue, fixture.frameSpec, bundle));
  CHECK(tuner.completed());
  CHECK(tuner.completionReason() ==
        alpakaTune::TunerCompletionReason::noValidConfiguration);
  CHECK_THROWS_AS(tuner.bestCandidateIndex(), std::logic_error);
  CHECK(tuner.history().size() == 1u);
}

TEST_CASE("pre-launch constraint rejection does not enter execution history",
          "[validity][constraints]") {
  auto selected = hostDevice();
  if (!selected)
    SKIP("The host backend is unavailable.");
  auto fixture = Fixture{*selected};
  auto config = fixedConfig(false);
  config.maximumConsecutiveStrategyRetries = 1u;
  auto const tunables = alpakaTune::constrain(
      alpakaTune::TunableBundle{qualityLevel(alpakaTune::RVals{1})},
      alpakaTune::restrict(
          qualityLevel,
          [](alpaka::concepts::VectorOrScalar auto const &) { return false; }));
  auto tuner = alpakaTune::makeTuner(config, tunables, fixture.device,
                                     "validity-pre-launch-rejection");
  auto const bundle = alpaka::KernelBundle{
      WriteKernel{}, fixture.output.getMdSpan(), qualityLevel};

  CHECK_THROWS(tuner.enqueue(fixture.queue, fixture.frameSpec, bundle));
  CHECK(tuner.history().empty());
  CHECK(tuner.lastCandidateIndex() == std::numeric_limits<std::size_t>::max());
  CHECK_THROWS_AS(tuner.lastConfig(), std::logic_error);
  CHECK(tuner.info().restrictionRejectedCount == 1u);
  CHECK(tuner.info().userInvalidatedCandidateCount == 0u);
}

#if ALPAKA_TUNE_HAS_JSON
TEST_CASE("user invalidation is excluded from compact persistence",
          "[validity][persistence]") {
  auto selected = hostDevice();
  if (!selected)
    SKIP("The host backend is unavailable.");
  auto fixture = Fixture{*selected};
  auto const directory = std::filesystem::temp_directory_path() /
                         "alpakaTune-user-invalidation-test";
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  auto config = fixedConfig(false);
  config.history.file = directory / "history.json";
  config.history.write = true;
  config.completeHistory.file = directory / "complete-history.json";
  config.completeHistory.write = true;
  auto const tunables =
      alpakaTune::TunableBundle{qualityLevel(alpakaTune::RVals{1, 2})};
  auto tuner = alpakaTune::makeTuner(config, tunables, fixture.device,
                                     "validity-persistence");
  auto const bundle = alpaka::KernelBundle{
      WriteKernel{}, fixture.output.getMdSpan(), qualityLevel};

  tuner.enqueue(fixture.queue, fixture.frameSpec, bundle);
  tuner.lastConfig().valid = false;
  tuner.enqueue(fixture.queue, fixture.frameSpec, bundle);
  alpakaTune::flushPersistence();

  auto compact = nlohmann::json{};
  auto complete = nlohmann::json{};
  std::ifstream{*config.history.file} >> compact;
  std::ifstream{*config.completeHistory.file} >> complete;
  REQUIRE(compact.at("contexts").size() == 1u);
  REQUIRE(complete.at("contexts").size() == 1u);
  auto const &compactContext = compact.at("contexts").begin().value();
  auto const &completeContext = complete.at("contexts").begin().value();
  CHECK(compactContext.at("configurations").size() == 1u);
  CHECK(compactContext.at("configurations")[0u]
            .at("configuration")
            .at("qualityLevel") == 2);
  CHECK(completeContext.at("user_invalidated_candidates") ==
        nlohmann::json::array({true, false}));
  CHECK_FALSE(completeContext.at("candidate_samples")[0u].empty());

  auto replayConfig = config;
  replayConfig.exploration = alpakaTune::ExplorationPolicy::offline;
  replayConfig.selection = alpakaTune::SelectionPolicy::fixed;
  replayConfig.maximumExecutions.reset();
  replayConfig.history.read = false;
  replayConfig.history.write = false;
  replayConfig.completeHistory.read = true;
  replayConfig.completeHistory.write = false;
  auto replay = alpakaTune::makeTuner(replayConfig, tunables, fixture.device,
                                      "validity-persistence");
  replay.enqueue(fixture.queue, fixture.frameSpec, bundle);
  CHECK(replay.loadedFromCache());
  CHECK(replay.lastConfig().candidateIndex == 1u);
  CHECK(replay.info().userInvalidatedCandidateCount == 1u);

  std::filesystem::remove_all(directory);
}
#endif
