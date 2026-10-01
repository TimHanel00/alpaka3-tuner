// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>
#include <alpakaTune/alpakaTune.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace workflowTests {
using Index = alpaka::Vec<std::size_t, 1u>;
inline constexpr auto batchSize = ALPAKA_TUNE_TUNABLE("batchSize");
inline constexpr auto unroll = ALPAKA_TUNE_TUNABLE("unroll");
inline constexpr auto integrationSteps =
    ALPAKA_TUNE_TUNABLE("integrationSteps");

void require(bool condition, std::string const &message) {
  if (!condition)
    throw std::runtime_error{message};
}

auto configFor(std::filesystem::path const &directory, std::string const &name,
               bool replay) -> alpakaTune::TunerConfig {
  auto config = alpakaTune::TunerConfig{};
  config.mode = replay ? alpakaTune::TuningMode::offline
                       : alpakaTune::TuningMode::onlineFixed;
  config.replayFastPath = true;
  config.strategy = alpakaTune::StrategyKind::exhaustive;
  config.queue = alpakaTune::QueueConfig{.warmupRuns = 1u,
                                         .noiseCancellationWindow = 4u,
                                         .maxConsecutiveRuns = 3u};
  config.runsPerCandidate = 2u;
  config.minimumRunsPerCandidate = 2u;
  config.mannWhitneyEarlyStop = false;
  if (replay)
    config.maximumExecutions.reset();
  else
    config.maximumExecutions = 512u;
  config.maximumRetiredConfigurations.reset();
  config.history = {.read = replay, .write = !replay};
  config.completeHistory = {.read = false, .write = !replay};
  if (!directory.empty()) {
    config.history.file = directory / (name + "-compact.json");
    config.completeHistory.file = directory / (name + "-complete.json");
  }
  return config;
}

auto readFile(std::filesystem::path const &path) -> std::string {
  std::ifstream stream{path};
  require(stream.good(),
          "Missing persisted workflow history: " + path.string());
  return {std::istreambuf_iterator<char>{stream}, {}};
}

struct Transform {
  ALPAKA_FN_ACC void operator()(auto const &acc,
                                alpaka::concepts::IMdSpan auto input,
                                alpaka::concepts::IMdSpan auto output,
                                Index extent, int bias, int batch,
                                auto lanes) const {
    auto const groups = Index{(extent.x() + batch - 1u) / batch};
    for (auto group :
         alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid,
                                   alpaka::IdxRange{groups})) {
      for (int offset = 0; offset < batch; offset += decltype(lanes)::value)
        for (int lane = 0; lane < decltype(lanes)::value; ++lane) {
          auto const index = group.x() * batch + offset + lane;
          if (offset + lane < batch && index < extent.x())
            output[index] = 3 * input[index] + bias;
        }
    }
  }
};

struct FinishTransform {
  ALPAKA_FN_ACC void operator()(auto const &acc,
                                alpaka::concepts::IMdSpan auto input,
                                alpaka::concepts::IMdSpan auto output,
                                Index extent, int bias) const {
    for (auto index :
         alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid,
                                   alpaka::IdxRange{extent}))
      output[index] = 2 * (input[index] - bias);
  }
};

void pipeline(auto device, auto executor,
              std::filesystem::path const &directory, bool replay) {
  constexpr Index extent{131071u};
  auto queue = alpakaTune::makeQueue(device, alpaka::queueKind::nonBlocking,
                                     alpakaTune::timing::enabled);
  auto input = alpaka::onHost::allocHost<int>(extent);
  auto result = alpaka::onHost::allocHostLike(input);
  // Alternating allocations force terminal replay to rebind each kernel's
  // views.
  auto inputA = alpaka::onHost::allocLike(device, input);
  auto inputB = alpaka::onHost::allocLike(device, input);
  auto intermediateA = alpaka::onHost::allocLike(device, input);
  auto intermediateB = alpaka::onHost::allocLike(device, input);
  auto outputA = alpaka::onHost::allocLike(device, input);
  auto outputB = alpaka::onHost::allocLike(device, input);
  auto frame = alpaka::onHost::FrameSpec{Index{2u}, Index{32u}, executor};
  auto const geometry = alpakaTune::TunableBundle{
      alpakaTune::frameExtent(
          alpakaTune::RVals<Index>{std::vector<Index>{Index{32u}, Index{64u}}}),
      alpakaTune::numFrames(
          alpakaTune::RVals<Index>{std::vector<Index>{Index{1u}, Index{3u}}})};
  auto const transformTunables =
      alpakaTune::TunableBundle{geometry, batchSize(alpakaTune::RVals{1, 7}),
                                unroll(alpakaTune::CVals<1, 4>{})};
  auto first = alpakaTune::makeTuner(configFor(directory, "transform", replay),
                                     transformTunables, device, executor,
                                     "workflow-transform");
  auto second =
      alpakaTune::makeTuner(configFor(directory, "finish", replay), geometry,
                            device, executor, "workflow-finish");
  std::size_t terminalRuns = 0u;
  for (std::size_t step = 0u; step < 96u; ++step) {
    int const bias = static_cast<int>(step) + (replay ? 10000 : 100);
    for (std::size_t index = 0u; index < extent.x(); ++index)
      input[index] = static_cast<int>(index % 997u) + bias;
    auto &source = step % 2u == 0u ? inputA : inputB;
    auto &intermediate = step % 2u == 0u ? intermediateA : intermediateB;
    auto &output = step % 2u == 0u ? outputA : outputB;
    alpaka::onHost::memcpy(queue, source, input);
    // Warm-ups and repeated candidate launches overwrite, rather than compound,
    // the application result. Both kernels use one queue for ordered
    // dependencies.
    bool const terminal = first.completed() && second.completed();
    first.enqueue(queue, frame,
                  alpaka::KernelBundle{Transform{}, source.getMdSpan(),
                                       intermediate.getMdSpan(), extent, bias,
                                       batchSize, unroll});
    second.enqueue(queue, frame,
                   alpaka::KernelBundle{FinishTransform{},
                                        intermediate.getMdSpan(),
                                        output.getMdSpan(), extent, bias});
    alpaka::onHost::memcpy(queue, result, output);
    alpaka::onHost::wait(queue);
    for (std::size_t index = 0u; index < extent.x(); ++index)
      if (result[index] != 6 * input[index])
        require(false, "Pipeline mismatch at step " + std::to_string(step) +
                           ", element " + std::to_string(index));
    if (terminal) {
      ++terminalRuns;
      require(!first.lastConfig().measured && !second.lastConfig().measured,
              "Completed pipeline still measures terminal replay");
    }
  }
  require(first.completed() && second.completed(), "Pipeline did not complete");
  require(terminalRuns >= 16u, "Insufficient pipeline replay coverage");
  require(first.info().candidateCount == 16u &&
              second.info().candidateCount == 4u,
          "Pipeline candidate Cartesian product changed");
  require(first.info().measuredCandidateCount == 16u &&
              second.info().measuredCandidateCount == 4u,
          "Pipeline failed to exercise every candidate");
  if (!replay) {
    for (std::size_t candidate = 0u; candidate < 16u; ++candidate)
      require(first.candidateRuntimeSamples(candidate).size() == 2u,
              "Transform sample budget or terminal replay changed");
    for (std::size_t candidate = 0u; candidate < 4u; ++candidate)
      require(second.candidateRuntimeSamples(candidate).size() == 2u,
              "Finish sample budget or terminal replay changed");
  }
  require(first.loadedFromCache() == replay &&
              second.loadedFromCache() == replay,
          "Pipeline persistence provenance mismatch");
}

struct HeatStep {
  ALPAKA_FN_ACC void operator()(auto const &acc,
                                alpaka::concepts::IMdSpan auto input,
                                alpaka::concepts::IMdSpan auto output,
                                Index extent, std::size_t width,
                                std::size_t height) const {
    for (auto index :
         alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid,
                                   alpaka::IdxRange{extent})) {
      auto const i = index.x();
      auto const x = i % width;
      auto const y = i / width;
      output[i] = x == 0u || y == 0u || x + 1u == width || y + 1u == height
                      ? input[i]
                      : input[i] + 0.125 * (input[i - 1u] + input[i + 1u] +
                                            input[i - width] +
                                            input[i + width] - 4.0 * input[i]);
    }
  }
};

void stencil(auto device, auto executor) {
  constexpr std::size_t width = 257u;
  constexpr std::size_t height = 193u;
  constexpr Index extent{width * height};
  auto queue = alpakaTune::makeQueue(device, alpaka::queueKind::nonBlocking,
                                     alpakaTune::timing::enabled);
  auto host = alpaka::onHost::allocHost<double>(extent);
  std::vector<double> reference(extent.x());
  std::vector<double> next(extent.x());
  for (std::size_t i = 0u; i < extent.x(); ++i)
    host[i] = reference[i] = static_cast<double>((i * 17u) % 101u) / 101.0;
  auto input = alpaka::onHost::allocLike(device, host);
  auto output = alpaka::onHost::allocLike(device, host);
  auto frame = alpaka::onHost::FrameSpec{Index{3u}, Index{32u}, executor};
  auto const tunables = alpakaTune::TunableBundle{
      alpakaTune::frameExtent(alpakaTune::RVals<Index>{
          std::vector<Index>{Index{32u}, Index{64u}, Index{128u}}}),
      alpakaTune::numFrames(
          alpakaTune::RVals<Index>{std::vector<Index>{Index{1u}, Index{5u}}})};
  auto tuner = alpakaTune::makeTuner(configFor({}, "heat", false), tunables,
                                     device, executor, "workflow-heat");
  alpaka::onHost::memcpy(queue, input, host);
  for (std::size_t step = 0u; step < 64u; ++step) {
    // Independent row/column reference, including fixed boundary conditions.
    for (std::size_t y = 0u; y < height; ++y)
      for (std::size_t x = 0u; x < width; ++x) {
        auto const i = y * width + x;
        next[i] = reference[i];
        if (x > 0u && y > 0u && x + 1u < width && y + 1u < height)
          next[i] += 0.125 * (reference[i - 1u] + reference[i + 1u] +
                              reference[i - width] + reference[i + width] -
                              4.0 * reference[i]);
      }
    tuner.enqueue(queue, frame,
                  alpaka::KernelBundle{HeatStep{}, input.getMdSpan(),
                                       output.getMdSpan(), extent, width,
                                       height});
    alpaka::onHost::memcpy(queue, host, output);
    alpaka::onHost::wait(queue);
    for (std::size_t i = 0u; i < extent.x(); ++i)
      if (!std::isfinite(host[i]) || std::abs(host[i] - next[i]) >= 1.0e-12)
        require(false, "Heat mismatch at step " + std::to_string(step) +
                           ", cell " + std::to_string(i));
    std::swap(input, output);
    reference.swap(next);
  }
  require(tuner.completed() && tuner.info().measuredCandidateCount == 6u,
          "Heat workflow did not exhaust its tuning candidates");
}

struct Integrate {
  ALPAKA_FN_ACC void operator()(auto const &acc,
                                alpaka::concepts::IMdSpan auto output,
                                Index extent, int steps) const {
    for (auto index :
         alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid,
                                   alpaka::IdxRange{extent})) {
      double const upper = 0.5 + 1.5 * index.x() / (extent.x() - 1u);
      double const delta = upper / steps;
      double sum = 0.0;
      for (int i = 0; i < steps; ++i) {
        double const x = (i + 0.5) * delta;
        sum += 1.0 / (1.0 + x * x);
      }
      output[index] = delta * sum;
    }
  }
};

void qualityWorkflow(auto device, auto executor) {
  constexpr Index extent{1025u};
  auto queue = alpakaTune::makeQueue(device, alpaka::queueKind::nonBlocking,
                                     alpakaTune::timing::disabled);
  auto host = alpaka::onHost::allocHost<double>(extent);
  auto output = alpaka::onHost::allocLike(device, host);
  auto frame = alpaka::onHost::FrameSpec{Index{4u}, Index{32u}, executor};
  auto config = configFor({}, "integration", false);
  config.queue.reset();
  auto const tunables = alpakaTune::TunableBundle{
      integrationSteps(alpakaTune::RVals{8, 64, 512})};
  auto tuner = alpakaTune::makeTuner(
      config, tunables, device, executor, "workflow-quadrature",
      alpakaTune::customMetric("seconds_per_valid_batch"));
  std::size_t validBatches = 0u;
  for (std::size_t batch = 0u; batch < 32u; ++batch) {
    auto const start = std::chrono::steady_clock::now();
    tuner.enqueue(queue, frame,
                  alpaka::KernelBundle{Integrate{}, output.getMdSpan(), extent,
                                       integrationSteps});
    alpaka::onHost::memcpy(queue, host, output);
    alpaka::onHost::wait(queue);
    bool valid = true;
    for (std::size_t i = 0u; i < extent.x(); ++i) {
      double const upper = 0.5 + 1.5 * i / (extent.x() - 1u);
      valid = valid && std::isfinite(host[i]) &&
              std::abs(host[i] - std::atan(upper)) < 1.0e-5;
    }
    tuner.provideMetric(
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count());
    if (!valid)
      tuner.lastConfig().valid = false;
    else
      ++validBatches;
    if (tuner.completed())
      require(valid && tuner.lastCandidateIndex() != 0u,
              "Quality workflow replayed a rejected approximation");
  }
  require(tuner.completed() && validBatches >= 16u,
          "Quality workflow did not reach valid terminal replay");
  require(tuner.info().measuredCandidateCount == 2u,
          "Quality workflow failed to measure both valid approximations");
  require(tuner.info().userInvalidatedCandidateCount == 1u,
          "Quality workflow did not reject the inaccurate approximation");
}

void run(auto device, auto executor, std::string const &workflow,
         std::filesystem::path const &directory, bool replay) {
  if (workflow == "pipeline")
    pipeline(device, executor, directory, replay);
  else if (workflow == "stencil")
    stencil(device, executor);
  else if (workflow == "quality")
    qualityWorkflow(device, executor);
  else
    throw std::runtime_error{"Unknown workflow"};
}
} // namespace workflowTests

int main(int argc, char **argv) {
  using namespace workflowTests;
  try {
    require(argc >= 2, "Usage: workflows pipeline|stencil|quality "
                       "[stage|replay directory] [--cuda]");
    std::string const workflow{argv[1]};
    bool const cuda = std::string{argv[argc - 1]} == "--cuda";
    int const argumentCount = argc - (cuda ? 1 : 0);
    require(argumentCount == 2 || argumentCount == 4,
            "Invalid workflow arguments");
    if (argumentCount == 4)
      require(workflow == "pipeline" && (std::string{argv[2]} == "stage" ||
                                         std::string{argv[2]} == "replay"),
              "Persistence requires pipeline stage|replay and a directory");
    bool const replay = argumentCount == 4 && std::string{argv[2]} == "replay";
    std::filesystem::path const directory = argumentCount == 4 ? argv[3] : "";
    std::vector<std::pair<std::filesystem::path, std::string>> saved;
    if (!directory.empty()) {
      if (replay) {
        for (auto const name : {"transform", "finish"})
          for (auto const suffix : {"-compact.json", "-complete.json"}) {
            auto const path = directory / (std::string{name} + suffix);
            saved.emplace_back(path, readFile(path));
          }
      } else {
        std::filesystem::remove_all(directory);
        std::filesystem::create_directories(directory);
      }
    }
    if (cuda) {
#if defined(ALPAKA_CMAKE_TARGET_CUDA)
      auto selector =
          alpaka::onHost::makeDeviceSelector(alpaka::onHost::DeviceSpec{
              alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu});
      require(selector.isAvailable(), "Requested CUDA device unavailable");
      run(selector.makeDevice(0u), alpaka::exec::gpuCuda, workflow, directory,
          replay);
#else
      throw std::runtime_error{"CUDA was not compiled into this executable"};
#endif
    } else {
      auto selector =
          alpaka::onHost::makeDeviceSelector(alpaka::onHost::DeviceSpec{
              alpaka::api::host, alpaka::deviceKind::cpu});
      require(selector.isAvailable(), "Host device unavailable");
      run(selector.makeDevice(0u), alpaka::exec::cpuSerial, workflow, directory,
          replay);
    }
    alpakaTune::flushPersistence();
    for (auto const &[path, contents] : saved)
      require(readFile(path) == contents, "Read-only replay modified history");
    std::cout << workflow << " workflow passed\n";
    return 0;
  } catch (std::exception const &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
