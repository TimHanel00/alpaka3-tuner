// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0
#include <alpaka/alpaka.hpp>
#include <alpakaTune/alpakaTune.hpp>

#include <algorithm>
#include <chrono>
#include <concepts>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

using Index = alpaka::Vec<std::uint32_t, 1u>;
inline constexpr auto batchSize = ALPAKA_TUNE_TUNABLE("benefitBatchSize");

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
                                   alpaka::IdxRange{groups}))
      for (std::uint32_t offset = 0u; offset < size; ++offset) {
        auto const index = group.x() * size + offset;
        if (index < extent.x())
          output[index] = input[index] + 1u;
      }
  }
};

struct Result {
  double wallSeconds{};
  std::uint64_t measurements{};
  double estimatedCost{};
};

auto run(std::string const &mode, std::uint64_t launches, std::uint32_t size,
         bool phaseChange, std::uint32_t repetition) -> Result {
  auto device = alpaka::onHost::makeDeviceSelector(
                    alpaka::onHost::DeviceSpec{alpaka::api::host,
                                               alpaka::deviceKind::cpu})
                    .makeDevice(0u);
  auto queue =
      device.makeQueue(alpaka::queueKind::nonBlocking, alpaka::timing::enabled);
  Index extent{size};
  auto input = alpaka::onHost::allocHost<std::uint32_t>(extent);
  auto result = alpaka::onHost::allocHostLike(input);
  auto deviceInput = alpaka::onHost::allocLike(device, input);
  auto deviceResult = alpaka::onHost::allocLike(device, result);
  for (std::uint32_t i = 0u; i < size; ++i)
    input[i] = i;
  alpaka::onHost::memcpy(queue, deviceInput, input);
  alpaka::onHost::wait(queue);
  auto frame = alpaka::onHost::FrameSpec{Index{1u}, Index{256u},
                                         alpaka::exec::cpuSerial};
  Result measured;
  auto const start = std::chrono::steady_clock::now();
  if (mode == "fixed") {
    for (std::uint64_t launch = 0u; launch < launches; ++launch) {
      auto current = Index{phaseChange && launch < launches / 2u ? 257u : size};
      queue.enqueue(
          frame, alpaka::KernelBundle{AddOne{}, deviceInput.getMdSpan(),
                                      deviceResult.getMdSpan(), current, 64u});
    }
  } else {
    auto config = alpakaTune::TunerConfig{};
    config.maximumExecutions = 32u;
    config.history = {.read = false, .write = false};
    config.completeHistory = {.read = false, .write = false};
    if (mode == "budget")
      config.budget = alpakaTune::BudgetConfig{.expectedLaunches = launches};
    auto tuner = alpakaTune::makeTuner(
        config,
        alpakaTune::TunableBundle{
            batchSize(alpakaTune::RVals{1u, 8u, 64u, 256u})},
        device, mode, size, launches, phaseChange, repetition);
    for (std::uint64_t launch = 0u; launch < launches; ++launch) {
      auto current = Index{phaseChange && launch < launches / 2u ? 257u : size};
      tuner.enqueue(queue, frame,
                    alpaka::KernelBundle{AddOne{}, deviceInput.getMdSpan(),
                                         deviceResult.getMdSpan(), current,
                                         batchSize});
      if (tuner.lastConfig().measured)
        ++measured.measurements;
    }
    // Include completion, diagnostics and tuner destruction in total time.
    alpaka::onHost::wait(queue);
    if (auto const budget = tuner.info().budget)
      measured.estimatedCost = budget->spentSeconds;
  }
  alpaka::onHost::memcpy(queue, result, deviceResult);
  alpaka::onHost::wait(queue);
  measured.wallSeconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  for (std::uint32_t i = 0u; i < size; ++i)
    if (result[i] != input[i] + 1u)
      throw std::runtime_error{"Incorrect application result"};
  return measured;
}

int main(int argc, char **argv) {
  auto const repetitions =
      argc > 1 ? static_cast<std::uint32_t>(std::stoul(argv[1])) : 5u;
  std::cout << "scenario,repetition,mode,launches,elements,wall_seconds,"
               "measurements,estimated_cost_seconds\n";
  for (auto const launches : {32u, 2000u})
    for (auto const size : {257u, 262147u})
      for (auto const shift : {false, true}) {
        if (shift && size == 257u)
          continue;
        std::string modes[]{"fixed", "adaptive", "budget"};
        for (std::uint32_t repeat = 0u; repeat < repetitions; ++repeat)
          for (std::uint32_t offset = 0u; offset < 3u; ++offset) {
            auto const &mode = modes[(repeat + offset) % 3u];
            auto const result = run(mode, launches, size, shift, repeat);
            std::cout << (shift ? "phase_change" : "steady") << ',' << repeat
                      << ',' << mode << ',' << launches << ',' << size << ','
                      << result.wallSeconds << ',' << result.measurements << ','
                      << result.estimatedCost << '\n';
          }
      }
}
