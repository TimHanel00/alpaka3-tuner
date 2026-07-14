// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>
#include <alpakaTune/alpakaTune.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

inline constexpr auto runtimeOffset = ALPAKA_TUNE_NAME("runtimeOffset");
inline constexpr auto simdWidth = ALPAKA_TUNE_NAME("simdWidth");
inline constexpr auto numBlocks = ALPAKA_TUNE_NAME("numBlocks");

constexpr std::size_t elementCount = 32'768u;
constexpr std::size_t launchCount = 10'000u;
constexpr std::size_t runsPerCandidate = 250u;
constexpr std::array<std::uint32_t, 2u> runtimeOffsets{0u, 1u};
constexpr std::array<std::size_t, 4u> simdWidths{1u, 2u, 4u, 8u};
constexpr std::array<std::size_t, 5u> blockCounts{1u, 2u, 4u, 8u, 16u};

static_assert(launchCount ==
              runtimeOffsets.size() * simdWidths.size() * blockCounts.size() * runsPerCandidate);

struct CandidateConfiguration {
  std::uint32_t offset;
  std::size_t width;
  std::size_t blocks;
};

[[nodiscard]] auto configurationFor(std::size_t candidate) -> CandidateConfiguration {
  auto const block = blockCounts.at(candidate % blockCounts.size());
  candidate /= blockCounts.size();
  return {runtimeOffsets.at(candidate / simdWidths.size()),
          simdWidths.at(candidate % simdWidths.size()), block};
}

struct VectorAddKernel {
  ALPAKA_FN_ACC void operator()(auto const &acc,
                                alpaka::concepts::IMdSpan auto const left,
                                alpaka::concepts::IMdSpan auto const right,
                                alpaka::concepts::IMdSpan auto output,
                                auto const &count, std::uint32_t offset,
                                auto width) const {
    auto grid = alpaka::onAcc::SimdAlgo{alpaka::onAcc::worker::threadsInGrid};
    grid.concurrent<decltype(width)::value>(
        acc, count,
        [offset](auto const &, auto &&a, auto &&b, auto &&c) {
          auto result = a.load() + b.load();
          result += offset;
          c = result;
        },
        left, right, output);
  }
};

[[nodiscard]] auto writeConfiguration(std::filesystem::path const &outputDirectory)
    -> std::filesystem::path {
  auto const persistenceDirectory = outputDirectory / "tuning-cache";
  std::filesystem::remove_all(persistenceDirectory);
  auto const configuration = outputDirectory / "session.yaml";
  std::ofstream yaml{configuration};
  if (!yaml)
    throw std::runtime_error{"Unable to create the instrumentation YAML file."};
  yaml << R"(schema_version: 1
tuning:
  strategy: bayesian_optimization
  random_seed: 0
  warmup_runs: 0
  runs_per_candidate: )"
       << runsPerCandidate << R"(
  minimum_runs_per_candidate: )"
       << runsPerCandidate << R"(
  ci_check_interval: 10
  ci_z_score: 2.576
  ci_relative_width: 0.05
  outlier_mad_scale: 3.5
  # Keep the equal 250-sample benchmark budget for every configuration.
  mann_whitney_early_stop: false
  mann_whitney_min_samples: 8
  mann_whitney_alpha: 0.05
  noise_cancellation_window: 3
  max_consecutive_runs: 3
persistence:
  directory: )"
       << std::quoted(persistenceDirectory.string()) << '\n';
  return configuration;
}

auto run(std::filesystem::path outputDirectory) -> int {
  using Data = std::uint32_t;
  using Index = alpaka::Vec<std::size_t, 1u>;

  outputDirectory = std::filesystem::absolute(std::move(outputDirectory));
  std::filesystem::create_directories(outputDirectory);
  auto const backend = alpaka::onHost::DeviceSpec{alpaka::api::host,
                                                   alpaka::deviceKind::cpu};
  auto selector = alpaka::onHost::makeDeviceSelector(backend);
  if (!selector.isAvailable()) {
    std::cerr << "The CpuSerial backend is unavailable.\n";
    return EXIT_FAILURE;
  }

  auto device = selector.makeDevice(0u);
  auto queue = device.makeQueue();
  auto const executor = alpaka::exec::cpuOmpBlocks;
  auto const frameSpec = alpaka::onHost::FrameSpec{
      Index{1u}, Index{elementCount}, executor};
  auto hostLeft = alpaka::onHost::allocHost<Data>(Index{elementCount});
  auto hostRight = alpaka::onHost::allocHostLike(hostLeft);
  auto hostOutput = alpaka::onHost::allocHostLike(hostLeft);
  for (std::size_t index = 0u; index < elementCount; ++index) {
    hostLeft[index] = static_cast<Data>(index);
    hostRight[index] = static_cast<Data>(elementCount - index);
  }

  auto deviceLeft = alpaka::onHost::allocLike(device, hostLeft);
  auto deviceRight = alpaka::onHost::allocLike(device, hostRight);
  auto deviceOutput = alpaka::onHost::allocLike(device, hostOutput);
  alpaka::onHost::memcpy(queue, deviceLeft, hostLeft);
  alpaka::onHost::memcpy(queue, deviceRight, hostRight);

  auto const configuration = writeConfiguration(outputDirectory);
  auto const tunables = alpakaTune::Tunables{
      alpakaTune::named(runtimeOffset, alpakaTune::RVals{0u, 1u}),
      alpakaTune::named(simdWidth, alpakaTune::CVals<1u, 2u, 4u, 8u>{}),
      alpakaTune::named(numBlocks,
                        alpakaTune::RVals<Index>{std::vector<Index>{
                            Index{1u}, Index{2u}, Index{4u}, Index{8u}, Index{16u}}})};
  auto context = alpakaTune::contextBuilder(configuration).createContextWith(
      tunables, device, alpaka::deviceKind::cpu, alpaka::api::host, executor,
      "instrumented-vector-add");
  if (context.strategyKind() != alpakaTune::StrategyKind::bayesianOptimization)
    throw std::logic_error{"The instrumentation YAML did not select Bayesian optimization."};
  auto const prototype = alpaka::KernelBundle{
      VectorAddKernel{}, deviceLeft, deviceRight, deviceOutput,
      Index{elementCount}, alpakaTune::markTunable(runtimeOffset),
      alpakaTune::markTunable(simdWidth)};

  auto csv = std::ofstream{outputDirectory / "samples.csv"};
  if (!csv)
    throw std::runtime_error{"Unable to create the instrumentation CSV file."};
  csv << "time_seconds,runtime_microseconds,candidate_index,runtime_offset,simd_width,num_blocks,"
         "candidate_raw_mean_microseconds,candidate_estimate_microseconds,"
         "accepted_samples\n";

  std::array<double, runtimeOffsets.size() * simdWidths.size() * blockCounts.size()> totals{};
  std::array<std::size_t, runtimeOffsets.size() * simdWidths.size() * blockCounts.size()> counts{};
  auto const benchmarkStart = std::chrono::steady_clock::now();
  for (std::size_t launch = 0u; launch < launchCount; ++launch) {
    auto const start = std::chrono::steady_clock::now();
    context.tune(queue, frameSpec, prototype);
    alpaka::onHost::wait(queue);
    auto const stop = std::chrono::steady_clock::now();
    auto const candidate = context.lastCandidateIndex();
    auto const candidateConfiguration = configurationFor(candidate);
    auto const runtime = std::chrono::duration<double, std::micro>{stop - start}.count();
    auto const time = std::chrono::duration<double>{stop - benchmarkStart}.count();
    totals.at(candidate) += runtime;
    auto const candidateMean = totals.at(candidate) / static_cast<double>(++counts.at(candidate));
    auto const statistics = context.candidateRuntimeStatistics(candidate);
    auto const candidateEstimate = statistics.estimate() * 1.0e6;
    csv << std::setprecision(12) << time << ',' << runtime << ',' << candidate << ','
        << candidateConfiguration.offset << ',' << candidateConfiguration.width << ','
        << candidateConfiguration.blocks << ','
        << candidateMean << ',' << candidateEstimate << ','
        << statistics.acceptedSampleCount << '\n';
  }

  if (!context.isTuningComplete())
    throw std::logic_error{"The 10,000-launch instrumentation budget did not finish tuning."};

  auto const lastConfiguration = configurationFor(context.lastCandidateIndex());
  alpaka::onHost::memcpy(queue, hostOutput, deviceOutput);
  alpaka::onHost::wait(queue);
  auto const expected = static_cast<Data>(elementCount + lastConfiguration.offset);
  for (std::size_t index = 0u; index < elementCount; ++index) {
    if (hostOutput[index] != expected)
      throw std::runtime_error{"The vector-add result did not match the launched runtime value."};
  }

  auto const winner = context.bestCandidateIndex();
  auto const winnerConfiguration = configurationFor(winner);
  auto const winnerStatistics = context.candidateRuntimeStatistics(winner);
  auto summary = std::ofstream{outputDirectory / "summary.txt"};
  if (!summary)
    throw std::runtime_error{"Unable to create the instrumentation summary file."};
  summary << "launches=" << launchCount << '\n'
          << "strategy=" << alpakaTune::strategyName(context.strategyKind()) << '\n'
          << "best_candidate_index=" << winner << '\n'
          << "runtime_offset=" << winnerConfiguration.offset << '\n'
          << "simd_width=" << winnerConfiguration.width << '\n'
          << "num_blocks=" << winnerConfiguration.blocks << '\n'
          << "raw_mean_runtime_microseconds=" << std::setprecision(12)
          << winnerStatistics.rawMean * 1.0e6 << '\n'
          << "robust_median_runtime_microseconds=" << std::setprecision(12)
          << winnerStatistics.estimate() * 1.0e6 << '\n'
          << "accepted_samples=" << winnerStatistics.acceptedSampleCount << '\n'
          << "measured_samples=" << winnerStatistics.sampleCount << '\n';
  std::cout << "Wrote " << (outputDirectory / "samples.csv") << '\n'
            << "Best configuration after " << launchCount << " launches: runtimeOffset="
            << winnerConfiguration.offset << ", simdWidth=" << winnerConfiguration.width
            << ", numBlocks=" << winnerConfiguration.blocks
            << ", robust median runtime=" << std::setprecision(6)
            << winnerStatistics.estimate() * 1.0e6 << " us\n";
  return EXIT_SUCCESS;
}

} // namespace

auto main(int argc, char **argv) -> int {
  try {
    auto const outputDirectory = argc > 1 ? std::filesystem::path{argv[1]}
                                          : std::filesystem::temp_directory_path() /
                                                "alpakaTune-vector-add-10000";
    return run(outputDirectory);
  } catch (std::exception const &error) {
    std::cerr << "instrumented vector-add failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
