// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0
#include <algorithm>
#include <alpaka/alpaka.hpp>
#include <alpakaTune/alpakaTune.hpp>
#include <concepts>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>

using Index = alpaka::Vec<std::uint32_t, 1u>;
inline constexpr auto batchSize = ALPAKA_TUNE_TUNABLE("batchSize");

struct AddOne {
  ALPAKA_FN_ACC void
  operator()(alpaka::onAcc::concepts::Acc auto const &acc,
             alpaka::concepts::IMdSpan auto input,
             alpaka::concepts::IMdSpan auto output, Index extent,
             std::convertible_to<std::uint32_t> auto batch) const {
    auto const size = static_cast<std::uint32_t>(batch);
    for (auto start : alpaka::onAcc::makeIdxMap(
             acc, alpaka::onAcc::worker::threadsInGrid,
             alpaka::IdxRange{Index{0u}, extent, Index{size}}))
      for (auto index = start.x();
           index < start.x() + size && index < extent.x(); ++index)
        output[index] = input[index] + 1u;
  }
};

int main(int argc, char **argv) {
  auto selector = alpaka::onHost::makeDeviceSelector(
      alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu});
  auto device = selector.makeDevice(0u);
  auto queue =
      device.makeQueue(alpaka::queueKind::nonBlocking, alpaka::timing::enabled);
  constexpr Index extent{262147u}; // Deliberately not a multiple of a batch.
  auto result = alpaka::onHost::allocHost<std::uint32_t>(extent);
  auto deviceInput = alpaka::onHost::allocLike(device, result);
  auto deviceResult = alpaka::onHost::allocLike(device, result);
  alpaka::onHost::fill(queue, deviceInput, 42u);

  auto config = alpakaTune::TunerConfig{};
  config.exploration = alpakaTune::ExplorationPolicy::online;
  config.selection = alpakaTune::SelectionPolicy::fixed;
  config.runsPerCandidate = 3u;
  config.minimumRunsPerCandidate = 3u;
  config.maximumExecutions = 100u;
  config.replayFastPath = true;
  if (argc == 2)
    config = alpakaTune::TunerConfig::fromYaml(argv[1]);

  auto frame = alpaka::onHost::FrameSpec{Index{4u}, Index{256u},
                                         alpaka::exec::cpuSerial};
  auto batchSizes = alpakaTune::RVals{32u, 64u, 128u};
#ifdef ALPAKA_TUNE_TUTORIAL_COMPILE_TIME
  auto candidates = batchSize(alpakaTune::CVals<32u, 64u, 128u>{});
#else
  auto candidates = batchSize(batchSizes);
#endif
  auto tunables = alpakaTune::TunableBundle{candidates};
  auto tuner =
      alpakaTune::makeTuner(config, tunables, device, alpaka::exec::cpuSerial,
                            "add-one-v1", extent.x());
  auto bundle =
      alpaka::KernelBundle{AddOne{}, deviceInput.getMdSpan(),
                           deviceResult.getMdSpan(), extent, batchSize};
  for (std::size_t launch = 0u; launch < 20u; ++launch) {
    alpaka::onHost::fill(queue, deviceResult, 0u);
    tuner.enqueue(queue, frame, bundle);
    // Replay may be asynchronous: wait on the queue that produced the data.
    alpaka::onHost::memcpy(queue, result, deviceResult);
    alpaka::onHost::wait(queue);
    if (!std::ranges::all_of(std::span{result.data(), extent.x()},
                             [](auto value) { return value == 43u; }))
      throw std::runtime_error{"Incorrect tutorial result"};
  }
  auto const info = tuner.info();
  std::cout << "20 launches checked\n" << std::fixed << std::setprecision(3);
  if (auto const runtimes = tuner.executionRuntimeSummary()) {
    std::cout << "Kernel runtimes (us; " << runtimes->sampleCount
              << " timed launches): min=" << runtimes->minimumSeconds * 1e6
              << "; max=" << runtimes->maximumSeconds * 1e6
              << "; median=" << runtimes->medianSeconds * 1e6
              << "; avg=" << runtimes->averageSeconds * 1e6 << '\n'
              << "Minimum-runtime configuration: batchSize="
              << batchSizes.at(runtimes->minimumExecution.candidateIndex)
              << '\n';
  } else {
    std::cout << "No launch runtimes recorded (unmeasured replay)\n";
  }
  std::cout << "Tuning: "
            << (info.tuningComplete ? "complete" : "still exploring")
            << "; history: "
            << (info.loadedFromCache ? "restored saved measurements"
                                     : "no saved measurements restored")
            << '\n';
}
