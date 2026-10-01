// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>
#include <alpakaTune/alpakaTune.hpp>

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <optional>
#include <string_view>
#include <thread>
#include <utility>

#if ALPAKA_TUNE_HAS_JSON
#include <nlohmann/json.hpp>
#endif

namespace {

inline constexpr auto qualityLevel = ALPAKA_TUNE_TUNABLE("qualityLevel");

template <typename T>
concept ProvidesCustomMetric = requires(T &tuner) { tuner.provideMetric(1.0); };

struct WriteKernel {
  ALPAKA_FN_ACC void operator()(auto const &acc,
                                alpaka::concepts::IMdSpan auto output,
                                int value) const {
    static_cast<void>(acc);
    output[0u] = value;
  }
};

[[nodiscard]] auto fixedConfig(std::size_t runs = 1u)
    -> alpakaTune::TunerConfig {
  auto config = alpakaTune::TunerConfig{};
  config.mode = alpakaTune::TuningMode::onlineFixed;
  config.strategy = alpakaTune::StrategyKind::exhaustive;
  config.queue.reset();
  config.runsPerCandidate = runs;
  config.minimumRunsPerCandidate = runs;
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
      alpakaTune::timing::disabled));
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
                                    alpakaTune::timing::disabled)),
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

TEST_CASE(
    "untimed custom metric launches finish before fixture buffers are released",
    "[metric][lifetime]") {
  auto selected = hostDevice();
  REQUIRE(selected.has_value());
  std::promise<void> finished;
  auto completion = finished.get_future();
  {
    auto fixture = Fixture{*selected};
    // Model a busy asynchronous executor: the kernel remains pending when
    // this scope exits. ASan catches a freed span target without the drain.
    fixture.queue.enqueueHostFn(
        [] { std::this_thread::sleep_for(std::chrono::milliseconds{10}); });
    auto const tunables =
        alpakaTune::TunableBundle{qualityLevel(alpakaTune::RVals{1})};
    auto tuner = alpakaTune::makeTuner(
        fixedConfig(), tunables, fixture.device,
        alpakaTune::customMetric("lifetime_cost"), "custom-metric-lifetime");
    tuner.enqueue(fixture.queue, fixture.frameSpec,
                  alpaka::KernelBundle{
                      WriteKernel{}, fixture.output.getMdSpan(), qualityLevel});
    tuner.provideMetric(1.0);
    fixture.queue.enqueueHostFn([&finished] { finished.set_value(); });
  }
  CHECK(completion.wait_for(std::chrono::milliseconds{0}) ==
        std::future_status::ready);
}

TEST_CASE("custom metrics replace compile-time timing instrumentation",
          "[metric]") {
  auto selected = hostDevice();
  if (!selected)
    SKIP("The host backend is unavailable.");
  auto fixture = Fixture{*selected};
  auto const tunables =
      alpakaTune::TunableBundle{qualityLevel(alpakaTune::RVals{1, 2})};
  auto tuner = alpakaTune::makeTuner(fixedConfig(2u), tunables, fixture.device,
                                     alpakaTune::customMetric("quality_loss"),
                                     "custom-metric-direct");
  static_assert(decltype(tuner)::usesCustomMetric);
  static_assert(ProvidesCustomMetric<decltype(tuner)>);
  auto timingTuner = alpakaTune::makeTuner(
      fixedConfig(2u), tunables, fixture.device, "timing-metric-type");
  static_assert(!decltype(timingTuner)::usesCustomMetric);
  static_assert(!ProvidesCustomMetric<decltype(timingTuner)>);
  auto const bundle = alpaka::KernelBundle{
      WriteKernel{}, fixture.output.getMdSpan(), qualityLevel};

  CHECK(tuner.metricKind() == alpakaTune::TuningMetricKind::custom);
  CHECK(tuner.metricName() == "quality_loss");
  CHECK_FALSE(tuner.info().runtimeMeasurementSource);
  CHECK_THROWS(tuner.provideMetric(1.0));
  CHECK_THROWS_AS(alpakaTune::customMetric(""), std::invalid_argument);

  while (!tuner.completed()) {
    tuner.enqueue(fixture.queue, fixture.frameSpec, bundle);
    auto const candidate = tuner.lastConfig().candidateIndex;
    CHECK_FALSE(tuner.lastConfig().runtimeSeconds);
    CHECK_FALSE(tuner.lastConfig().metricValue);
    CHECK_FALSE(tuner.lastConfig().measured);
    if (tuner.history().size() == 1u)
      CHECK_THROWS_AS(tuner.provideMetric(-1.0), std::invalid_argument);
    if (tuner.history().size() == 2u)
      CHECK_THROWS_AS(
          tuner.provideMetric(std::numeric_limits<double>::infinity()),
          std::invalid_argument);
    tuner.provideMetric(candidate == 0u ? 4.0 : 1.0);
    CHECK(tuner.lastConfig().metricValue ==
          std::optional<double>{candidate == 0u ? 4.0 : 1.0});
    CHECK(tuner.lastConfig().measured);
    CHECK_THROWS(tuner.provideMetric(2.0));
  }

  REQUIRE(tuner.history().size() == 4u);
  CHECK(tuner.bestCandidateIndex() == 1u);
  CHECK(tuner.candidateMetricSamples(0u).size() == 2u);
  CHECK(tuner.candidateMetricSamples(1u).size() == 2u);
  CHECK(tuner.info().metricName == "quality_loss");
  CHECK(tuner.info().metricKind == alpakaTune::TuningMetricKind::custom);
  CHECK_THROWS_AS(tuner.provideMetric(1.0), std::logic_error);
}

TEST_CASE("an omitted custom metric rejects exactly the preceding candidate",
          "[metric][scheduler]") {
  auto selected = hostDevice();
  if (!selected)
    SKIP("The host backend is unavailable.");
  auto fixture = Fixture{*selected};
  auto const tunables =
      alpakaTune::TunableBundle{qualityLevel(alpakaTune::RVals{1, 2, 3})};
  auto tuner = alpakaTune::makeTuner(
      fixedConfig(), tunables, fixture.device,
      alpakaTune::customMetric("application_cost"), "missing-metric");
  auto const bundle = alpaka::KernelBundle{
      WriteKernel{}, fixture.output.getMdSpan(), qualityLevel};

  tuner.enqueue(fixture.queue, fixture.frameSpec, bundle);
  REQUIRE(tuner.lastConfig().candidateIndex == 0u);

  tuner.enqueue(fixture.queue, fixture.frameSpec, bundle);
  CHECK(tuner.lastConfig().candidateIndex == 1u);
  CHECK(tuner.info().missingMetricCandidateCount == 1u);
  CHECK(tuner.info().restrictionRejectedCount == 0u);
  CHECK(tuner.info().userInvalidatedCandidateCount == 0u);
  CHECK_FALSE(tuner.history().front().metricValue);
  tuner.provideMetric(2.0);

  tuner.enqueue(fixture.queue, fixture.frameSpec, bundle);
  CHECK(tuner.lastConfig().candidateIndex == 2u);
  tuner.provideMetric(3.0);

  CHECK(tuner.completed());
  CHECK(tuner.bestCandidateIndex() == 1u);
  CHECK(tuner.history().size() == 3u);
  CHECK(tuner.candidateMetricSamples(0u).empty());
  CHECK(tuner.info().measuredCandidateCount == 2u);
}

TEST_CASE("a sole missing custom metric terminates without a winner",
          "[metric]") {
  auto selected = hostDevice();
  if (!selected)
    SKIP("The host backend is unavailable.");
  auto fixture = Fixture{*selected};
  auto const tunables =
      alpakaTune::TunableBundle{qualityLevel(alpakaTune::RVals{1})};
  auto tuner = alpakaTune::makeTuner(
      fixedConfig(), tunables, fixture.device,
      alpakaTune::customMetric("application_cost"), "missing-only-metric");
  auto const bundle = alpaka::KernelBundle{
      WriteKernel{}, fixture.output.getMdSpan(), qualityLevel};

  tuner.enqueue(fixture.queue, fixture.frameSpec, bundle);
  CHECK_THROWS(tuner.enqueue(fixture.queue, fixture.frameSpec, bundle));
  CHECK(tuner.completed());
  CHECK(tuner.completionReason() ==
        alpakaTune::TunerCompletionReason::noValidConfiguration);
  CHECK(tuner.info().missingMetricCandidateCount == 1u);
  CHECK_THROWS_AS(tuner.bestCandidateIndex(), std::logic_error);
}

#if ALPAKA_TUNE_HAS_JSON
TEST_CASE("custom metric identity and missing candidates persist",
          "[metric][persistence]") {
  auto selected = hostDevice();
  if (!selected)
    SKIP("The host backend is unavailable.");
  auto fixture = Fixture{*selected};
  auto const directory =
      std::filesystem::temp_directory_path() / "alpakaTune-custom-metric-test";
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  auto config = fixedConfig();
  config.history.file = directory / "history.json";
  config.history.write = true;
  config.completeHistory.file = directory / "complete-history.json";
  config.completeHistory.write = true;
  auto const tunables =
      alpakaTune::TunableBundle{qualityLevel(alpakaTune::RVals{1, 2})};
  auto tuner = alpakaTune::makeTuner(
      config, tunables, fixture.device,
      alpakaTune::customMetric("joules_per_result"), "metric-persistence");
  auto const bundle = alpaka::KernelBundle{
      WriteKernel{}, fixture.output.getMdSpan(), qualityLevel};

  tuner.enqueue(fixture.queue, fixture.frameSpec, bundle);
  tuner.enqueue(fixture.queue, fixture.frameSpec, bundle);
  tuner.provideMetric(7.0);
  alpakaTune::flushPersistence();

  auto compact = nlohmann::json{};
  auto complete = nlohmann::json{};
  std::ifstream{*config.history.file} >> compact;
  std::ifstream{*config.completeHistory.file} >> complete;
  auto const &compactContext = compact.at("contexts").begin().value();
  auto const &completeContext = complete.at("contexts").begin().value();
  CHECK(compactContext.at("metric") ==
        nlohmann::json{{"kind", "custom"}, {"name", "joules_per_result"}});
  CHECK(completeContext.at("metadata").at("metric") ==
        compactContext.at("metric"));
  CHECK(completeContext.at("missing_metric_candidates") ==
        nlohmann::json::array({true, false}));
  REQUIRE(compactContext.at("configurations").size() == 1u);
  CHECK(compactContext.at("configurations")[0u].at("median_metric_value") ==
        7.0);

  auto replayConfig = config;
  replayConfig.mode = alpakaTune::TuningMode::offline;
  replayConfig.maximumExecutions.reset();
  replayConfig.history.read = true;
  replayConfig.history.write = false;
  replayConfig.completeHistory.read = false;
  replayConfig.completeHistory.write = false;
  auto replay = alpakaTune::makeTuner(
      replayConfig, tunables, fixture.device, "metric-persistence",
      alpakaTune::customMetric("joules_per_result"));
  replay.enqueue(fixture.queue, fixture.frameSpec, bundle);
  CHECK(replay.loadedFromCache());
  CHECK(replay.lastConfig().candidateIndex == 1u);
  replay.provideMetric(6.0);
  CHECK(replay.lastConfig().metricValue == std::optional<double>{6.0});
  CHECK_FALSE(replay.lastConfig().measured);

  auto wrongMetric = alpakaTune::makeTuner(
      replayConfig, tunables, fixture.device,
      alpakaTune::customMetric("different_metric"), "metric-persistence");
  CHECK_THROWS_AS(wrongMetric.enqueue(fixture.queue, fixture.frameSpec, bundle),
                  std::runtime_error);

  std::filesystem::remove_all(directory);
}
#endif
