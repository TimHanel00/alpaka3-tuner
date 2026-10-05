// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>

#include <tuning.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numbers>

namespace {

inline constexpr auto termCount = ALPAKA_TUNE_TUNABLE("termCount");
inline constexpr auto candidateTerms =
    std::array<std::uint32_t, 4u>{1'000u, 10'000u, 100'000u, 1'000'000u};

/** Approximate pi with the Leibniz series using one tunable amount of work. */
struct ApproximatePiKernel {
  ALPAKA_FN_ACC void operator()(auto const &acc,
                                alpaka::concepts::IMdSpan auto output,
                                std::uint32_t terms) const {
    static_cast<void>(acc);
    auto sum = 0.0;
    for (std::uint32_t term = 0u; term < terms; ++term) {
      auto const sign = term % 2u == 0u ? 1.0 : -1.0;
      sum += sign / static_cast<double>(2u * term + 1u);
    }
    output[0u] = 4.0 * sum;
  }
};

[[nodiscard]] auto tuningConfig() -> alpakaTune::TunerConfig {
  auto config = alpakaTune::TunerConfig{};
  config.exploration = alpakaTune::ExplorationPolicy::online;
  config.selection = alpakaTune::SelectionPolicy::fixed;
  config.strategy = alpakaTune::StrategyKind::exhaustive;
  config.queue.reset();
  config.runsPerCandidate = 3u;
  config.minimumRunsPerCandidate = 3u;
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

auto run(auto const &backend) -> int {
  auto selector =
      alpaka::onHost::makeDeviceSelector(alpaka::onHost::DeviceSpec{backend});
  if (!selector.isAvailable())
    return EXIT_SUCCESS;
  auto device = selector.makeDevice(0u);
  auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);
  auto tuningQueue = alpakaTune::makeQueue(
      device, alpaka::queueKind::nonBlocking, alpakaTune::timing::enabled);
  auto hostResult = alpaka::onHost::allocHost<double>(1u);
  auto deviceResult = alpaka::onHost::allocLike(device, hostResult);
  auto const frameSpec =
      alpaka::onHost::FrameSpec{1u, 1u, alpaka::getExecutor(backend)};
  auto const tunables = alpakaTune::TunableBundle{
      termCount(alpakaTune::RVals<std::uint32_t>::list(
          candidateTerms[0u], candidateTerms[1u], candidateTerms[2u],
          candidateTerms[3u]))};
  auto tuner = alpakaTune::makeTuner(tuningConfig(), tunables, device, backend,
                                     "post-evaluation-pi");
  auto const bundle = alpaka::KernelBundle{ApproximatePiKernel{},
                                           deviceResult.getMdSpan(), termCount};

  constexpr auto maximumError = 2.0e-5;
  while (!tuner.completed()) {
    tuner.enqueue(tuningQueue, frameSpec, bundle);
    alpaka::onHost::memcpy(queue, hostResult, deviceResult);
    alpaka::onHost::wait(queue);

    auto &executed = tuner.lastConfig();
    auto const error = std::abs(hostResult[0u] - std::numbers::pi);
    executed.valid = error <= maximumError;
    std::cout << "terms=" << candidateTerms.at(executed.candidateIndex)
              << " error=" << error
              << " valid=" << static_cast<bool>(executed.valid) << '\n';
  }

  auto const winner = tuner.bestCandidateIndex();
  std::cout << "fastest valid term count: " << candidateTerms.at(winner)
            << " after " << tuner.history().size() << " launches\n";
  return EXIT_SUCCESS;
}

} // namespace

auto main(int argc, char **argv) -> int {
  if (!alpakaTune::consumeBackendOptions(argc, argv))
    return EXIT_FAILURE;
  return alpaka::onHost::executeForEach(
      [](alpaka::concepts::BackendSpec auto const &backend) {
        if (!alpakaTune::backendSelected(backend))
          return EXIT_SUCCESS;
        return run(backend);
      },
      alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs,
                                  alpaka::exec::enabledExecutors));
}
