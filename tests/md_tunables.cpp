// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>
#include <alpakaTune/alpakaTune.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <set>
#include <type_traits>
#include <utility>

namespace {
inline constexpr auto tile = ALPAKA_TUNE_TUNABLE("tile");
inline constexpr auto workers = ALPAKA_TUNE_TUNABLE("workers");
using Vec1 = alpaka::Vec<std::size_t, 1u>;
using Vec2 = alpaka::Vec<std::size_t, 2u>;
using Vec3 = alpaka::Vec<std::size_t, 3u>;
using Small = alpaka::CVec<std::size_t, 8u, 2u>;
using Large = std::integer_sequence<std::size_t, 16u, 4u>;

struct WriteShape {
  ALPAKA_FN_ACC void operator()(auto const &,
                                alpaka::concepts::IMdSpan auto output,
                                alpaka::concepts::Vector auto shape,
                                std::uint32_t workerCount) const {
    auto encoded = std::uint32_t{};
    for (std::size_t d{}; d < ALPAKA_TYPEOF(shape)::dim(); ++d)
      encoded = encoded * 100u + static_cast<std::uint32_t>(shape[d]);
    output[0u] = encoded * 10u + workerCount;
  }
};

// A Cartesian implementation would fail compilation for the two mixed shapes.
struct WriteListedCompileShape {
  ALPAKA_FN_ACC void operator()(auto const &acc,
                                alpaka::concepts::IMdSpan auto output,
                                alpaka::concepts::CVector auto shape,
                                std::uint32_t workerCount) const {
    static_assert(shape[0u] == shape[1u] * 4u);
    WriteShape{}(acc, output, shape, workerCount);
  }
};

struct EmptyKernel {
  ALPAKA_FN_ACC void operator()(auto const &) const {}
};

auto hostDevice() {
  return alpaka::onHost::makeDeviceSelector(
             alpaka::onHost::DeviceSpec{alpaka::api::host,
                                        alpaka::deviceKind::cpu})
      .makeDevice(0u);
}

auto configuration() {
  auto config = alpakaTune::TunerConfig{};
  config.exploration = alpakaTune::ExplorationPolicy::online;
  config.selection = alpakaTune::SelectionPolicy::fixed;
  config.strategy = alpakaTune::StrategyKind::exhaustive;
  config.runsPerCandidate = config.minimumRunsPerCandidate = 1u;
  config.mannWhitneyEarlyStop = false;
  config.maximumExecutions = 100u;
  config.maximumRetiredConfigurations.reset();
  config.queue.reset();
  config.history = {.read = false, .write = false};
  config.completeHistory = {.read = false, .write = false};
  return config;
}

auto runShapes(auto tunables, auto kernel, std::size_t candidateCount,
               std::size_t dimensionCount) {
  auto device = hostDevice();
  auto queue = alpakaTune::makeQueue(device, alpaka::queueKind::nonBlocking,
                                     alpakaTune::timing::disabled);
  auto output = alpaka::onHost::alloc<std::uint32_t>(device, Vec1{1u});
  auto launch =
      alpaka::onHost::FrameSpec{Vec1{1u}, Vec1{1u}, alpaka::exec::cpuSerial};
  auto tuner = alpakaTune::makeTuner(configuration(), tunables, device,
                                     alpakaTune::customMetric("cost"));
  CHECK(tuner.info().candidateCount == candidateCount);
  CHECK(tuner.candidateConfiguration(0u).size() == dimensionCount);
  auto shapes = std::set<std::uint32_t>{};
  for (std::size_t run{}; run < 100u && !tuner.completed(); ++run) {
    tuner.enqueue(
        queue, launch,
        alpaka::KernelBundle{kernel, output.getMdSpan(), tile, workers});
    alpaka::onHost::wait(queue);
    shapes.insert(output[0u]);
    tuner.provideMetric(static_cast<double>(output[0u]));
  }
  CHECK(tuner.completed());
  CHECK(tuner.info().restrictionRejectedCount == 0u);
  return shapes;
}
} // namespace

TEST_CASE("runtime MD policies preserve default component tuning") {
  auto choices = alpakaTune::RVals{Vec2{8u, 2u}, Vec2{16u, 4u}};
  auto scalar = workers(alpakaTune::RVals{1u, 2u});
  auto expectedIndependent = std::set<std::uint32_t>{
      8021u, 8022u, 8041u, 8042u, 16021u, 16022u, 16041u, 16042u};
  CHECK(runShapes(alpakaTune::TunableBundle{tile(choices), scalar},
                  WriteShape{}, 8u, 3u) == expectedIndependent);
  CHECK(runShapes(
            alpakaTune::TunableBundle{
                tile(choices, alpakaTune::mdPolicy::independent), scalar},
            WriteShape{}, 8u, 3u) == expectedIndependent);
  CHECK(runShapes(
            alpakaTune::TunableBundle{
                tile(choices, alpakaTune::mdPolicy::listed), scalar},
            WriteShape{}, 4u,
            2u) == std::set<std::uint32_t>{8021u, 8022u, 16041u, 16042u});
}

TEST_CASE("compile-time MD policies compile and launch the selected shapes") {
  auto choices = alpakaTune::CTypes<Small, Large>{};
  auto scalar = workers(alpakaTune::RVals{1u, 2u});
  auto expectedIndependent = std::set<std::uint32_t>{
      8021u, 8022u, 8041u, 8042u, 16021u, 16022u, 16041u, 16042u};
  CHECK(runShapes(alpakaTune::TunableBundle{tile(choices), scalar},
                  WriteShape{}, 8u, 3u) == expectedIndependent);
  CHECK(runShapes(
            alpakaTune::TunableBundle{
                tile(choices, alpakaTune::mdPolicy::independent), scalar},
            WriteShape{}, 8u, 3u) == expectedIndependent);
  CHECK(runShapes(
            alpakaTune::TunableBundle{
                tile(choices, alpakaTune::mdPolicy::listed), scalar},
            WriteListedCompileShape{}, 4u,
            2u) == std::set<std::uint32_t>{8021u, 8022u, 16041u, 16042u});
}

TEST_CASE("listed three-dimensional vectors combine with compiled scalars") {
  auto scalar = workers(alpakaTune::CVals<1u, 2u>{});
  auto expected = std::set<std::uint32_t>{802031u, 802032u, 1604051u, 1604052u};
  CHECK(runShapes(
            alpakaTune::TunableBundle{
                tile(alpakaTune::RVals{Vec3{8u, 2u, 3u}, Vec3{16u, 4u, 5u}},
                     alpakaTune::mdPolicy::listed),
                scalar},
            WriteShape{}, 4u, 2u) == expected);
  using Small3 = alpaka::CVec<std::size_t, 8u, 2u, 3u>;
  using Large3 = std::integer_sequence<std::size_t, 16u, 4u, 5u>;
  CHECK(runShapes(
            alpakaTune::TunableBundle{tile(alpakaTune::CTypes<Small3, Large3>{},
                                           alpakaTune::mdPolicy::listed),
                                      scalar},
            WriteListedCompileShape{}, 4u, 2u) == expected);
}

TEST_CASE("listed MD launch choices compose with coverage restrictions") {
  auto device = hostDevice();
  auto queue = alpakaTune::makeQueue(device, alpaka::queueKind::nonBlocking,
                                     alpakaTune::timing::disabled);
  auto frame = alpaka::onHost::FrameSpec{Vec2{4u, 4u}, Vec2{8u, 2u},
                                         alpaka::exec::cpuSerial};
  auto extents = alpakaTune::tuneFrameExtent(
      frame, alpakaTune::CTypes<Small, Large>{}, alpakaTune::mdPolicy::listed);
  auto counts = alpakaTune::tuneNumFrames(
      frame, alpakaTune::RVals{Vec2{4u, 4u}, Vec2{2u, 2u}},
      alpakaTune::mdPolicy::listed);
  auto tunables = alpakaTune::makeFrameSpecTuning(
      extents, counts, alpakaTune::preserveCoverage(frame));
  auto tuner = alpakaTune::makeTuner(configuration(), tunables, device,
                                     alpakaTune::customMetric("cost"));
  CHECK(tuner.info().candidateCount == 4u);
  for (std::size_t run{}; run < 10u && !tuner.completed(); ++run) {
    tuner.enqueue(queue, frame, alpaka::KernelBundle{EmptyKernel{}});
    tuner.provideMetric(1.0);
  }
  CHECK(tuner.completed());
  CHECK(tuner.info().measuredCandidateCount == 2u);
  CHECK(tuner.info().restrictionRejectedCount == 2u);
  alpaka::onHost::wait(queue);

  auto thread = alpaka::onHost::ThreadSpec{Vec2::fill(1u), Vec2::fill(1u),
                                           alpaka::exec::cpuSerial};
  auto threads = alpakaTune::tuneNumThreads(
      thread, alpakaTune::CTypes<alpaka::CVec<std::size_t, 1u, 1u>>{},
      alpakaTune::mdPolicy::listed);
  auto blocks = alpakaTune::tuneNumBlocks(
      thread, alpakaTune::RVals{Vec2{1u, 2u}, Vec2{2u, 1u}},
      alpakaTune::mdPolicy::listed);
  auto threadTuner = alpakaTune::makeTuner(
      configuration(), alpakaTune::TunableBundle{threads, blocks}, device,
      alpakaTune::customMetric("cost"));
  for (std::size_t run{}; run < 10u && !threadTuner.completed(); ++run) {
    threadTuner.enqueue(queue, thread, alpaka::KernelBundle{EmptyKernel{}});
    threadTuner.provideMetric(1.0);
  }
  CHECK(threadTuner.completed());
  CHECK(threadTuner.info().measuredCandidateCount == 2u);
  alpaka::onHost::wait(queue);
}

#if ALPAKA_TUNE_HAS_JSON
TEST_CASE("listed MD vectors replay from compact and complete histories") {
  auto device = hostDevice();
  auto queue = alpakaTune::makeQueue(device, alpaka::queueKind::nonBlocking,
                                     alpakaTune::timing::disabled);
  auto launch =
      alpaka::onHost::FrameSpec{Vec1{1u}, Vec1{1u}, alpaka::exec::cpuSerial};
  auto output = alpaka::onHost::alloc<std::uint32_t>(device, Vec1{1u});
  auto checkReplay = [&](auto choices, auto kernel, auto label) {
    auto bundle =
        alpaka::KernelBundle{kernel, output.getMdSpan(), tile, workers};
    auto tunables =
        alpakaTune::TunableBundle{tile(choices, alpakaTune::mdPolicy::listed),
                                  workers(alpakaTune::RVals{1u})};
    for (bool compact : {false, true}) {
      auto directory = std::filesystem::path{"md-tunables-history"} / label /
                       (compact ? "compact" : "complete");
      std::filesystem::create_directories(directory);
      auto config = configuration();
      if (compact)
        config.history = {
            .file = directory / "history.json", .read = false, .write = true};
      else
        config.completeHistory = {
            .file = directory / "history.json", .read = false, .write = true};
      {
        auto tuner = alpakaTune::makeTuner(config, tunables, device,
                                           alpakaTune::customMetric("cost"));
        for (std::size_t run{}; run < 10u && !tuner.completed(); ++run) {
          tuner.enqueue(queue, launch, bundle);
          alpaka::onHost::wait(queue);
          tuner.provideMetric(static_cast<double>(output[0u]));
        }
        REQUIRE(tuner.completed());
        alpakaTune::flushPersistence();
      }
      config.exploration = alpakaTune::ExplorationPolicy::offline;
      config.maximumExecutions.reset();
      config.history.read = compact;
      config.history.write = false;
      config.completeHistory.read = !compact;
      config.completeHistory.write = false;
      auto replay = alpakaTune::makeTuner(config, tunables, device,
                                          alpakaTune::customMetric("cost"));
      replay.enqueue(queue, launch, bundle);
      alpaka::onHost::wait(queue);
      CHECK(replay.loadedFromCache());
      CHECK(replay.completed());
      CHECK(output[0u] == 8021u);
      std::filesystem::remove_all(directory);
    }
  };
  checkReplay(alpakaTune::CTypes<Small, Large>{}, WriteListedCompileShape{},
              "compile-time");
  checkReplay(alpakaTune::RVals{Vec2{8u, 2u}, Vec2{16u, 4u}}, WriteShape{},
              "runtime");
}
#endif
