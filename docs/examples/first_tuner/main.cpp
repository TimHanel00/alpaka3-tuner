// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0
#include <alpaka/alpaka.hpp>
#include <alpakaTune/alpakaTune.hpp>
#include <concepts>
#include <cstdint>
#include <iostream>
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
    auto const groups = Index{(extent.x() + size - 1u) / size};
    for (auto group :
         alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid,
                                   alpaka::IdxRange{groups})) {
      for (std::uint32_t offset = 0u; offset < size; ++offset) {
        auto const index = group.x() * size + offset;
        if (index < extent.x())
          output[index] = input[index] + 1u;
      }
    }
  }
};

int main(int argc, char **argv) {
  auto selector = alpaka::onHost::makeDeviceSelector(
      alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu});
  auto device = selector.makeDevice(0u);
  auto queue =
      device.makeQueue(alpaka::queueKind::nonBlocking, alpaka::timing::enabled);
  constexpr Index extent{262147u}; // Deliberately not a multiple of a batch.
  auto input = alpaka::onHost::allocHost<std::uint32_t>(extent);
  auto result = alpaka::onHost::allocHostLike(input);
  auto deviceInput = alpaka::onHost::allocLike(device, input);
  auto deviceResult = alpaka::onHost::allocLike(device, result);
  for (std::uint32_t i = 0u; i < extent.x(); ++i)
    input[i] = i;
  alpaka::onHost::memcpy(queue, deviceInput, input);

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
#ifdef ALPAKA_TUNE_TUTORIAL_COMPILE_TIME
  auto candidates = batchSize(alpakaTune::CVals<32u, 64u, 128u>{});
#else
  auto candidates = batchSize(alpakaTune::RVals{32u, 64u, 128u});
#endif
  auto tunables = alpakaTune::TunableBundle{candidates};
  auto tuner =
      alpakaTune::makeTuner(config, tunables, device, alpaka::exec::cpuSerial,
                            "add-one-v1", extent.x());
  auto bundle =
      alpaka::KernelBundle{AddOne{}, deviceInput.getMdSpan(),
                           deviceResult.getMdSpan(), extent, batchSize};
  for (std::size_t launch = 0u; launch < 20u; ++launch) {
    tuner.enqueue(queue, frame, bundle);
    // Replay may be asynchronous: wait on the queue that produced the data.
    alpaka::onHost::memcpy(queue, result, deviceResult);
    alpaka::onHost::wait(queue);
    for (std::uint32_t i = 0u; i < extent.x(); ++i)
      if (result[i] != input[i] + 1u)
        throw std::runtime_error{"Incorrect tutorial result"};
  }
  std::cout << "20 launches checked; candidates=" << tuner.info().candidateCount
            << "; completed=" << tuner.completed()
            << "; loaded=" << tuner.loadedFromCache() << '\n';
}
