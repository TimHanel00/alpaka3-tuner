// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>
#include <alpakaTune/alpakaTune.hpp>

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#define NumberOfInsideSteps 1000u
#define NumberOfOutsideSteps 8u

namespace {

inline constexpr auto innerStrategy = ALPAKA_TUNE_TUNABLE("innerStrategy");
inline constexpr auto innerQueueWindow =
    ALPAKA_TUNE_TUNABLE("innerQueueWindow");
inline constexpr auto innerConsecutiveRuns =
    ALPAKA_TUNE_TUNABLE("innerConsecutiveRuns");

using Index = alpaka::Vec<std::size_t, 1u>;

struct VectorAddKernel {
  ALPAKA_FN_ACC void operator()(auto const &acc,
                                alpaka::concepts::IMdSpan auto inputA,
                                alpaka::concepts::IMdSpan auto inputB,
                                alpaka::concepts::IMdSpan auto output,
                                Index extent) const {
    for (auto index :
         alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid,
                                   alpaka::IdxRange{extent}))
      output[index] = inputA[index] + inputB[index];
  }
};

auto testDirectory() -> std::filesystem::path {
  auto const directory =
      std::filesystem::temp_directory_path() / "alpakaTune-tune-the-tuner-test";
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  return directory;
}

template <typename Device> class VectorAddTuner {
public:
  VectorAddTuner(Device device, std::filesystem::path directory,
                 alpakaTune::TunerConfig config)
      : m_device(std::move(device)),
        m_queue(alpakaTune::makeQueue(m_device, alpaka::queueKind::nonBlocking,
                                      alpakaTune::timing::enabled)),
        m_directory(std::move(directory)), m_config(std::move(config)),
        m_inputA(alpaka::onHost::allocHost<std::uint32_t>(extent)),
        m_inputB(alpaka::onHost::allocHost<std::uint32_t>(extent)),
        m_output(alpaka::onHost::allocHost<std::uint32_t>(extent)),
        m_frameSpec{Index{1u}, Index{32u}, alpaka::exec::cpuSerial},
        m_tunables{alpakaTune::frameExtent(alpakaTune::RVals<Index>{
            std::vector<Index>{Index{16u}, Index{32u}}})} {
    for (std::size_t index = 0u; index < extent.x(); ++index) {
      m_inputA[index] = static_cast<std::uint32_t>(index);
      m_inputB[index] = static_cast<std::uint32_t>(2u * index);
      m_output[index] = 0u;
    }
  }

  void tuneVectorAdd(alpakaTune::StrategyKind strategy, std::size_t queueWindow,
                     std::size_t consecutiveRuns) {
    m_configurations.emplace(strategy, queueWindow, consecutiveRuns);
    auto config = m_config;
    config.strategy = strategy;
    config.queue =
        alpakaTune::QueueConfig{.disable = false,
                                .warmupRuns = 0u,
                                .noiseCancellationWindow = queueWindow,
                                .maxConsecutiveRuns = consecutiveRuns};
    auto const name = std::string{alpakaTune::strategyName(strategy)} + "-" +
                      std::to_string(queueWindow) + "-" +
                      std::to_string(consecutiveRuns);
    config.history.file = m_directory / (name + "-history.json");
    config.completeHistory.file =
        m_directory / (name + "-complete-history.json");
    auto tuner = alpakaTune::makeTuner(
        config, m_tunables, m_device, alpaka::deviceKind::cpu,
        alpaka::api::host, alpaka::exec::cpuSerial, "vector-add/inner");
    auto const bundle = alpaka::KernelBundle{VectorAddKernel{}, m_inputA,
                                             m_inputB, m_output, extent};
    for (std::size_t step = 0u; step < NumberOfInsideSteps; ++step) {
      tuner.enqueue(m_queue, m_frameSpec, bundle);
      ++m_innerEnqueueCount;
    }
  }

  [[nodiscard]] auto innerEnqueueCount() const noexcept -> std::size_t {
    return m_innerEnqueueCount.load();
  }

  [[nodiscard]] auto configurationCount() const noexcept -> std::size_t {
    return m_configurations.size();
  }

  [[nodiscard]] auto resultIsCorrect() const -> bool {
    alpaka::onHost::wait(m_queue);
    for (std::size_t index = 0u; index < extent.x(); ++index)
      if (m_output[index] != m_inputA[index] + m_inputB[index])
        return false;
    return true;
  }

private:
  static constexpr Index extent{64u};

  Device m_device;
  ALPAKA_TYPEOF(alpakaTune::makeQueue(m_device, alpaka::queueKind::nonBlocking,
                                      alpakaTune::timing::enabled))
  m_queue;
  std::filesystem::path m_directory;
  alpakaTune::TunerConfig m_config;
  decltype(alpaka::onHost::allocHost<std::uint32_t>(extent)) m_inputA;
  decltype(alpaka::onHost::allocHost<std::uint32_t>(extent)) m_inputB;
  decltype(alpaka::onHost::allocHost<std::uint32_t>(extent)) m_output;
  decltype(alpaka::onHost::FrameSpec{Index{1u}, Index{32u},
                                     alpaka::exec::cpuSerial}) m_frameSpec;
  decltype(alpakaTune::TunableBundle{
      alpakaTune::frameExtent(alpakaTune::RVals<Index>{
          std::vector<Index>{Index{16u}, Index{32u}}})}) m_tunables;
  std::set<std::tuple<alpakaTune::StrategyKind, std::size_t, std::size_t>>
      m_configurations;
  std::atomic<std::size_t> m_innerEnqueueCount{0u};
};

struct TuneTheTunerKernel {
  template <typename Accelerator, typename InnerTuner, typename Strategy,
            typename QueueWindow, typename ConsecutiveRuns>
  ALPAKA_FN_ACC void operator()(Accelerator const &, InnerTuner *innerTuner,
                                Strategy strategy, QueueWindow queueWindow,
                                ConsecutiveRuns consecutiveRuns) const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    (void)innerTuner;
#else
    innerTuner->tuneVectorAdd(strategy, queueWindow, consecutiveRuns);
#endif
  }
};

} // namespace

TEST_CASE("a host tuner can tune the configuration of another tuner",
          "[integration][tuner-isolation]") {
  auto const backend =
      alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
  auto selector = alpaka::onHost::makeDeviceSelector(backend);
  REQUIRE(selector.isAvailable());

  auto device = selector.makeDevice(0u);
  auto queue = alpakaTune::makeQueue(device, alpaka::queueKind::nonBlocking,
                                     alpakaTune::timing::enabled);
  auto const outerFrameSpec =
      alpaka::onHost::FrameSpec{Index{1u}, Index{1u}, alpaka::exec::cpuSerial};
  auto const directory = testDirectory();
  auto const innerConfig = alpakaTune::TunerConfig{
      .mode = alpakaTune::TuningMode::onlineFixed,
      .queue = alpakaTune::QueueConfig{.disable = false,
                                       .warmupRuns = 0u,
                                       .noiseCancellationWindow = 1u,
                                       .maxConsecutiveRuns = 1u},
      .runsPerCandidate = 10u,
      .minimumRunsPerCandidate = 1u,
      .maximumExecutions = NumberOfInsideSteps,
      .strategy = alpakaTune::StrategyKind::exhaustive,
      .completeHistory = {.file = directory / "inner.json"}};
  VectorAddTuner innerTuner{device, directory, innerConfig};

  auto const outerTunables = alpakaTune::TunableBundle{
      innerStrategy(alpakaTune::RVals<alpakaTune::StrategyKind>{
          alpakaTune::StrategyKind::exhaustive,
          alpakaTune::StrategyKind::random}),
      innerQueueWindow(alpakaTune::RVals<std::size_t>{1u, 2u}),
      innerConsecutiveRuns(alpakaTune::RVals<std::size_t>{1u, 2u})};
  auto outerConfig = innerConfig;
  outerConfig.runsPerCandidate = 1u;
  outerConfig.maximumExecutions = NumberOfOutsideSteps;
  outerConfig.history.file = directory / "outer-history.json";
  outerConfig.completeHistory.file = directory / "outer-complete-history.json";
  auto outerTuner = alpakaTune::makeTuner(
      outerConfig, outerTunables, device, alpaka::deviceKind::cpu,
      alpaka::api::host, alpaka::exec::cpuSerial, "tune-the-tuner/outer");
  auto const outerBundle = alpaka::KernelBundle{
      TuneTheTunerKernel{}, &innerTuner, alpakaTune::markTunable(innerStrategy),
      alpakaTune::markTunable(innerQueueWindow),
      alpakaTune::markTunable(innerConsecutiveRuns)};

  for (std::size_t step = 0u; step < NumberOfOutsideSteps; ++step)
    outerTuner.enqueue(queue, outerFrameSpec, outerBundle);
  alpaka::onHost::wait(queue);

  CHECK(outerTuner.isTuningComplete());
  CHECK(outerTuner.info().executionCount == NumberOfOutsideSteps);
  CHECK(innerTuner.configurationCount() == NumberOfOutsideSteps);
  CHECK(innerTuner.innerEnqueueCount() ==
        NumberOfOutsideSteps * NumberOfInsideSteps);
  CHECK(innerTuner.resultIsCorrect());
}
