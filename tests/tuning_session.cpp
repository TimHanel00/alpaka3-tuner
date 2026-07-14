// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>
#include <alpakaTune/alpakaTune.hpp>

#include <cstdlib>
#include <concepts>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

namespace {

inline constexpr auto runtimeValue = ALPAKA_TUNE_NAME("runtimeValue");
inline constexpr auto compileTimeValue = ALPAKA_TUNE_NAME("compileTimeValue");
inline constexpr auto compileTimeOffset = ALPAKA_TUNE_NAME("compileTimeOffset");
inline constexpr auto numFrames = ALPAKA_TUNE_NAME("numFrames");

static_assert(std::same_as<alpakaTune::CValsRange<0, 4, 2>,
                           alpakaTune::CVals<0, 2, 4>>);

struct WriteKernel {
  ALPAKA_FN_ACC void operator()(auto const &acc,
                                alpaka::concepts::IMdSpan auto output,
                                int value) const {
    static_cast<void>(acc);
    output[0u] = value;
  }
};

struct CompileWriteKernel {
  ALPAKA_FN_ACC void operator()(auto const &acc,
                                alpaka::concepts::IMdSpan auto output,
                                auto value, auto offset) const {
    static_cast<void>(acc);
    output[0u] = decltype(value)::value + decltype(offset)::value;
  }
};

auto writeConfiguration() -> std::filesystem::path {
  auto const directory = std::filesystem::temp_directory_path() / "alpakaTune-context-test";
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  auto const configuration = directory / "session.yaml";
  std::ofstream output{configuration};
  output << R"(schema_version: 1
tuning:
  strategy: exhaustive
  random_seed: 0
  warmup_runs: 0
  runs_per_candidate: 2
  noise_cancellation_window: 3
  max_consecutive_runs: 1
persistence:
  directory: )" << (directory / "history").string() << '\n';
  return configuration;
}

} // namespace

auto main() -> int {
  using Index = alpaka::Vec<std::size_t, 1u>;
  auto const backend = alpaka::onHost::DeviceSpec{alpaka::api::host,
                                                   alpaka::deviceKind::cpu};
  auto selector = alpaka::onHost::makeDeviceSelector(backend);
  if (!selector.isAvailable())
    return EXIT_SUCCESS;
  auto device = selector.makeDevice(0u);
  auto queue = device.makeQueue();
  auto const executor = alpaka::exec::cpuSerial;
  auto const frameSpec = alpaka::onHost::FrameSpec{Index{1u}, Index{1u}, executor};
  auto host = alpaka::onHost::allocHost<int>(Index{1u});
  auto output = alpaka::onHost::allocLike(device, host);

  auto const configuration = writeConfiguration();
  auto const tunables = alpakaTune::Tunables{
      alpakaTune::named(runtimeValue, alpakaTune::RVals{1, 2, 3})};
  auto context = alpakaTune::contextBuilder(configuration)
                     .createContextWith(tunables, device, alpaka::deviceKind::cpu,
                                        alpaka::api::host, executor, "context-test");
  auto const bundle = alpaka::KernelBundle{
      WriteKernel{}, output.getMdSpan(), alpakaTune::markTunable(runtimeValue)};

  for (std::size_t launch = 0u; launch < 6u; ++launch)
    context.tune(queue, frameSpec, bundle);
  if (!context.isTuningComplete() || context.bestCandidateIndex() >= 3u)
    return EXIT_FAILURE;
  if (context.lastCandidateIndex() == std::numeric_limits<std::size_t>::max())
    return EXIT_FAILURE;

#if ALPAKA_TUNE_HAS_JSON
  auto cached = alpakaTune::contextBuilder(configuration)
                    .createContextWith(tunables, device, alpaka::deviceKind::cpu,
                                       alpaka::api::host, executor, "context-test");
  cached.tune(queue, frameSpec, bundle);
  alpaka::onHost::wait(queue);
  if (!cached.loadedFromCache() || !cached.isTuningComplete())
    return EXIT_FAILURE;
  alpaka::onHost::memcpy(queue, host, output);
  alpaka::onHost::wait(queue);
  if (host[0u] < 1 || host[0u] > 3)
    return EXIT_FAILURE;
#endif

  auto const compileTunables = alpakaTune::Tunables{
      alpakaTune::named(compileTimeValue, alpakaTune::CVals<1, 2>{}),
      alpakaTune::named(compileTimeOffset, alpakaTune::CVals<4, 8>{})};
  auto compileContext = alpakaTune::contextBuilder(configuration)
                            .createContextWith(compileTunables, device,
                                               alpaka::deviceKind::cpu,
                                               alpaka::api::host, executor,
                                               "compile-context-test");
  auto const compileBundle = alpaka::KernelBundle{
      CompileWriteKernel{}, output.getMdSpan(),
      alpakaTune::markTunable(compileTimeValue),
      alpakaTune::markTunable(compileTimeOffset)};
  for (std::size_t launch = 0u; launch < 8u; ++launch)
    compileContext.tune(queue, frameSpec, compileBundle);
  if (!compileContext.isTuningComplete() ||
      compileContext.bestCandidateIndex() >= 4u)
    return EXIT_FAILURE;

  // Reserved launch dimensions remain host-side: no KernelBundle marker is
  // required, yet the context rebuilds the FrameSpec for each candidate.
  auto const frameTunables = alpakaTune::Tunables{
      alpakaTune::named(numFrames,
                        alpakaTune::RVals<Index>{std::vector<Index>{Index{1u}}})};
  auto frameContext = alpakaTune::contextBuilder(configuration)
                          .createContextWith(frameTunables, device,
                                             alpaka::deviceKind::cpu,
                                             alpaka::api::host, executor,
                                             "frame-context-test");
  auto const frameBundle = alpaka::KernelBundle{WriteKernel{}, output.getMdSpan(), 7};
  for (std::size_t launch = 0u; launch < 2u; ++launch)
    frameContext.tune(queue, frameSpec, frameBundle);
  alpaka::onHost::wait(queue);
  if (!frameContext.isTuningComplete())
    return EXIT_FAILURE;

  std::filesystem::remove_all(configuration.parent_path());
  return EXIT_SUCCESS;
}
