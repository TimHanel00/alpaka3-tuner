// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>
#include <alpakaTune/alpakaTune.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace {

inline constexpr auto simdWidth = ALPAKA_TUNE_NAME("simdWidth");
constexpr std::size_t elementCount = 4096u;

struct VectorAddKernel {
  ALPAKA_FN_ACC void operator()(auto const &acc,
                                alpaka::concepts::IMdSpan auto const left,
                                alpaka::concepts::IMdSpan auto const right,
                                alpaka::concepts::IMdSpan auto output,
                                auto const &count, auto width) const {
    auto grid = alpaka::onAcc::SimdAlgo{alpaka::onAcc::worker::threadsInGrid};
    grid.concurrent<decltype(width)::value>(
        acc, count,
        [](auto const &, auto &&a, auto &&b, auto &&c) { c = a.load() + b.load(); },
        left, right, output);
  }
};

auto runBackend(auto const backend) -> int {
  using Data = std::uint32_t;
  using Index = alpaka::Vec<std::size_t, 1u>;
  auto selector =
      alpaka::onHost::makeDeviceSelector(alpaka::onHost::DeviceSpec{backend});
  if (!selector.isAvailable())
    return EXIT_SUCCESS;

  auto device = selector.makeDevice(0u);
  auto queue = device.makeQueue();
  Index const extent{elementCount};
  auto hostLeft = alpaka::onHost::allocHost<Data>(extent);
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

  auto const executor = alpaka::getExecutor(backend);
  auto const frameSpec =
      alpaka::onHost::FrameSpec{Index{16u}, Index{256u}, executor};
  auto const tunables = alpakaTune::Tunables{
      alpakaTune::named(simdWidth, alpakaTune::CVals<1u, 2u, 4u, 8u>{})};
  auto context = alpakaTune::contextBuilder().createContextWith(
      tunables, device, executor, "compile-time-vector-add");
  auto const prototype = alpaka::KernelBundle{
      VectorAddKernel{}, deviceLeft, deviceRight, deviceOutput, extent,
      alpakaTune::markTunable(simdWidth)};

  // The packaged default has one warm-up and one measured launch per variant.
  for (std::size_t launch = 0u; launch < 8u; ++launch)
    context.tune(queue, frameSpec, prototype);
  context.tune(queue, frameSpec, prototype); // enqueue the selected winner

  alpaka::onHost::memcpy(queue, hostOutput, deviceOutput);
  alpaka::onHost::wait(queue);
  for (std::size_t index = 0u; index < elementCount; ++index) {
    if (hostOutput[index] != elementCount) {
      std::cerr << "incorrect result at " << index << '\n';
      return EXIT_FAILURE;
    }
  }

  std::cout << "backend " << alpaka::onHost::demangledName(executor)
            << ": selected compile-time SIMD candidate "
            << context.bestCandidateIndex() << '\n';
  return EXIT_SUCCESS;
}

} // namespace

auto main() -> int {
  return alpaka::onHost::executeForEach(
      [](alpaka::concepts::BackendSpec auto const &backend) {
        return runBackend(backend);
      },
      alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs,
                                  alpaka::exec::enabledExecutors));
}
