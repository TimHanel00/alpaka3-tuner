// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#include ALPAKA_TUNE_HEAT_WINNERS_HEADER
#include <alpakaTune/BackendSelection.hpp>
#include <alpakaTune/alpakaTune.hpp>
#include <example/heatEquation/StencilVariants.hpp>
#include <heatEquation2D/src/BoundaryKernel.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#if ALPAKA_LANG_CUDA
#include <cuda_runtime.h>
#endif

#ifndef ALPAKA_TUNE_HEAT_SOURCE_REVISION
#define ALPAKA_TUNE_HEAT_SOURCE_REVISION "unrecorded"
#endif

namespace alpakaTune::benchmarks::heatEquation {
inline constexpr auto variantName = ALPAKA_TUNE_TUNABLE("stencilVariant");
using Clock = std::chrono::steady_clock;

struct Options {
  std::vector<Index> sizes;
  Index device{}, samples{9}, repetitions{31}, steps{64};
  std::uint64_t seed{20261008};
  bool selfTest{};
  std::string profile;
  std::optional<std::filesystem::path> csv, exportWinners;
};

auto number(std::string_view text, bool zero = false) -> Index {
  Index value{};
  auto const [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size() ||
      (!zero && value == 0u))
    throw std::invalid_argument{"Invalid integer: " + std::string{text}};
  return value;
}

auto parseOptions(int argc, char **argv) -> Options {
  Options options;
  for (int argument = 1; argument < argc; ++argument) {
    std::string_view const name{argv[argument]};
    auto value = [&]() -> std::string_view {
      if (++argument == argc)
        throw std::invalid_argument{std::string{name} + " needs a value"};
      return argv[argument];
    };
    if (name == "--size")
      options.sizes.push_back(number(value()));
    else if (name == "--device")
      options.device = number(value(), true);
    else if (name == "--samples-per-candidate")
      options.samples = number(value());
    else if (name == "--repetitions")
      options.repetitions = number(value());
    else if (name == "--steps")
      options.steps = number(value());
    else if (name == "--seed")
      options.seed = number(value(), true);
    else if (name == "--csv")
      options.csv = std::filesystem::path{value()};
    else if (name == "--export-winners")
      options.exportWinners = std::filesystem::path{value()};
    else if (name == "--profile")
      options.profile = value();
    else if (name == "--self-test")
      options.selfTest = true;
    else if (name == "--help") {
      std::cout << "FP64 heat equation: alpaka3 default and compiled tuning\n"
                   "--backend api:deviceKind --executor NAME --device INDEX\n"
                   "--size N (repeatable) --samples-per-candidate N\n"
                   "--repetitions N --steps N --seed N --csv FILE\n"
                   "--export-winners HEADER --self-test\n"
                   "Replay executable: --profile default|winner\n";
      std::exit(EXIT_SUCCESS);
    } else
      throw std::invalid_argument{"Unknown option: " + std::string{name}};
  }
  if (options.samples > 1000u || options.repetitions > 10000u ||
      options.steps > 100000u)
    throw std::invalid_argument{"Sample or step count is too large"};
  if (!options.profile.empty() && options.profile != "default" &&
      options.profile != "winner")
    throw std::invalid_argument{"--profile expects default or winner"};
  if (options.sizes.empty())
    options.sizes = options.selfTest
                        ? std::vector<Index>{16u, 31u, 64u, 67u}
                        : std::vector<Index>{128u,  256u,  512u,  1024u,
                                             2048u, 4096u, 8192u, 16384u};
  for (auto size : options.sizes)
    if (size > 16384u || (!options.selfTest && size % 16u != 0u))
      throw std::invalid_argument{"Benchmark sizes must be multiples of 16 "
                                  "and at most 16384"};
  if (options.selfTest && (!options.profile.empty() || options.exportWinners))
    throw std::invalid_argument{"Self-test cannot export or profile winners"};
  return options;
}

auto quote(std::string_view text) -> std::string {
  std::string result{"\""};
  for (auto character : text) {
    if (character == '"')
      result += '"';
    result += character;
  }
  return result + '"';
}
auto cppString(std::string_view text) -> std::string {
  std::string result{"\""};
  for (auto character : text) {
    if (character == '\\' || character == '"')
      result += '\\';
    result += character;
  }
  return result + '"';
}
auto median(std::vector<double> values) -> double {
  if (values.empty())
    throw std::runtime_error{"No timing samples"};
  std::ranges::sort(values);
  auto const middle = values.size() / 2u;
  return values.size() % 2u ? values[middle]
                            : (values[middle - 1u] + values[middle]) / 2.0;
}
auto interval(std::vector<double> const &baseline,
              std::vector<double> const &winner, std::uint64_t seed)
    -> std::array<double, 2> {
  if (baseline.size() != winner.size() || baseline.size() < 5u)
    throw std::invalid_argument{"At least five paired samples required"};
  std::mt19937_64 random{seed};
  std::uniform_int_distribution<std::size_t> index{0u, baseline.size() - 1u};
  std::vector<double> ratios;
  for (Index sample = 0; sample < 2000u; ++sample) {
    std::vector<double> first, second;
    for (std::size_t row = 0; row < baseline.size(); ++row) {
      auto const selected = index(random);
      first.push_back(baseline[selected]);
      second.push_back(winner[selected]);
    }
    ratios.push_back(median(first) / median(second));
  }
  std::ranges::sort(ratios);
  return {ratios[50], ratios[1949]};
}

struct Metadata {
  std::string device, architecture, backend, executor, uuid;
  Index sms{};
  std::int32_t driver{}, runtime{};
  Index ompThreads{};
};
auto metadata(alpaka::onHost::concepts::Device auto const &device,
              alpaka::concepts::Executor auto executor) -> Metadata {
  auto const properties = device.getDeviceProperties();
  Metadata result{device.getName(),
                  "host",
                  device.getApi().getName(),
                  alpaka::onHost::demangledName(executor),
                  "",
                  properties.multiProcessorCount,
                  0,
                  0};
#if defined(__x86_64__)
  if (result.backend == "Host")
    result.architecture = "x86_64";
#elif defined(__aarch64__)
  if (result.backend == "Host")
    result.architecture = "aarch64";
#endif
#if ALPAKA_OMP
  if constexpr (std::same_as<ALPAKA_TYPEOF(executor),
                             alpaka::exec::CpuOmpBlocks>)
    result.ompThreads = static_cast<Index>(omp_get_max_threads());
#endif
#if ALPAKA_LANG_CUDA
  if constexpr (std::same_as<ALPAKA_TYPEOF(device.getApi()),
                             alpaka::api::Cuda>) {
    cudaDeviceProp native{};
    if (cudaGetDeviceProperties(&native, device.getNativeHandle()) !=
        cudaSuccess)
      throw std::runtime_error{"Cannot query CUDA device properties"};
    result.architecture =
        "sm_" + std::to_string(native.major) + std::to_string(native.minor);
    cudaDriverGetVersion(&result.driver);
    cudaRuntimeGetVersion(&result.runtime);
    static constexpr char hex[] = "0123456789abcdef";
    for (auto byte : native.uuid.bytes) {
      auto const value = static_cast<unsigned char>(byte);
      result.uuid += hex[value >> 4u];
      result.uuid += hex[value & 15u];
    }
  }
#endif
  return result;
}
auto prefix(Metadata const &meta, Index size) -> std::string {
  return quote(meta.device) + ',' + quote(meta.architecture) + ',' +
         quote(meta.backend) + ',' + quote(meta.executor) + ',' +
         std::to_string(size) + ',';
}

struct Output {
  std::ofstream summary, samples, tuning, finalists;
  explicit Output(Options const &options) {
    if (!options.csv)
      return;
    auto const path = *options.csv;
    if (!path.parent_path().empty())
      std::filesystem::create_directories(path.parent_path());
    summary.open(path);
    samples.open(path.string() + ".samples.csv");
    tuning.open(path.string() + ".tuning.csv");
    finalists.open(path.string() + ".finalists.csv");
    if (!summary || !samples || !tuning || !finalists)
      throw std::runtime_error{"Cannot open CSV outputs"};
    std::string const identity = "device,architecture,backend,executor,size,";
    summary << identity
            << "mode,precision,catalog,legal_candidates,variant,grid_policy,"
               "implementation,threads_x,threads_y,rows,padding,blocks,steps,"
               "tuning_wall_s,default_step_s,winner_step_s,default_stencil_s,"
               "winner_stencil_s,speedup,ci_low,ci_high,default_wall_s,"
               "winner_wall_s\n";
    samples << identity
            << "mode,implementation,repetition,scope,queue_s,wall_s,steps\n";
    tuning
        << identity
        << "execution,variant,grid_policy,measured,batch,seconds_per_stencil\n";
    finalists << identity << "variant,grid_policy,repetition,batch,stencil_s\n";
    for (auto *stream : {&summary, &samples, &tuning, &finalists})
      *stream << std::setprecision(12);
  }
};

auto parameters(Index size) -> Parameters {
  double const dx = 1.0 / (double(size) + 1.0);
  return {size, dx, dx, 0.2 * dx * dx};
}
auto tileCount(Layout layout, Index size) -> Index {
  return alpaka::divCeil(size, layout.tileX()) *
         alpaka::divCeil(size, layout.tileY());
}
auto gridBlocks(Choice choice, Index size, Index sms) -> Index {
  auto const total = tileCount(layouts.at(choice.variant), size);
  return choice.gridPolicy == 0u
             ? total
             : std::min(total, std::max(1u, sms) *
                                   (choice.gridPolicy == 1u ? 2u : 8u));
}

auto originalFrame(Index size, alpaka::concepts::Executor auto executor) {
  return alpaka::onHost::FrameSpec{Vec{size / 16u, size / 16u},
                                   alpaka::CVec<Index, 16u, 16u>{}, executor};
}
auto originalBundle(alpaka::concepts::IBuffer auto const &input,
                    alpaka::concepts::IBuffer auto &output, Parameters p) {
  return alpaka::KernelBundle{::StencilKernel{},
                              input,
                              output,
                              alpaka::CVec<Index, 16u, 16u>{},
                              alpaka::CVec<Index, 18u, 18u>{},
                              Vec{p.size, p.size},
                              p.dx,
                              p.dy,
                              p.dt};
}

template <Index Variant>
void launch(std::integral_constant<Index, Variant>, Choice choice, Index sms,
            auto const &queue, alpaka::concepts::Executor auto executor,
            alpaka::concepts::IBuffer auto const &input,
            alpaka::concepts::IBuffer auto &output, Parameters p) {
  if constexpr (Variant == 0u) {
    queue.enqueue(originalFrame(p.size, executor),
                  originalBundle(input, output, p));
  } else {
    static constexpr auto layout = layouts[Variant];
    auto const threads = alpaka::exec::isSeqExecutor_v<ALPAKA_TYPEOF(executor)>
                             ? Vec{1u, 1u}
                             : Vec{layout.threadsY, layout.threadsX};
    queue.enqueue(
        alpaka::onHost::ThreadSpec{Vec{1u, gridBlocks(choice, p.size, sms)},
                                   threads, executor},
        alpaka::KernelBundle{TunedStencil{}, input, output, p,
                             std::integral_constant<Index, Variant>{}});
  }
}

#if !ALPAKA_TUNE_HEAT_REPLAY
template <Index Variant = 0u, typename Function>
decltype(auto) dispatch(Index variant, Function &&function) {
  if (variant == Variant)
    return function(std::integral_constant<Index, Variant>{});
  if constexpr (Variant + 1u < layouts.size())
    return dispatch<Variant + 1u>(variant, std::forward<Function>(function));
  else
    throw std::out_of_range{"Unknown stencil variant"};
}
#else
template <std::size_t Row = 0u, typename Function>
void dispatchSelected(Index size, Function &&function) {
  static constexpr auto selected = selectedWinners[Row];
  if (selected.size == size)
    function(std::integral_constant<Index, selected.variant>{},
             Choice{selected.variant, selected.gridPolicy});
  else if constexpr (Row + 1u < selectedWinners.size())
    dispatchSelected<Row + 1u>(size, std::forward<Function>(function));
  else
    throw std::invalid_argument{"No inserted winner for requested size"};
}
#endif

void boundary(auto const &queue, alpaka::concepts::Executor auto executor,
              alpaka::concepts::IBuffer auto &output, Parameters p,
              Index step) {
  auto const extent = p.size + 2u;
  queue.enqueue(alpaka::onHost::FrameSpec{alpaka::Vec{extent / 16u},
                                          alpaka::Vec{16u}, executor},
                alpaka::KernelBundle{::BoundaryKernel{}, output.getMdSpan(),
                                     Vec{extent, extent}, step, p.dx, p.dy,
                                     p.dt});
}

struct Duration {
  double queue, wall;
};
auto measure(alpaka::onHost::concepts::Device auto &device, auto const &queue,
             auto &&function) -> Duration {
  auto start = device.makeEvent(alpaka::timing::enabled);
  auto end = device.makeEvent(alpaka::timing::enabled);
  auto const wallStart = Clock::now();
  queue.enqueue(start);
  function();
  queue.enqueue(end);
  double const seconds = alpaka::onHost::getElapsedTime(start, end).count();
  double const wall =
      std::chrono::duration<double>(Clock::now() - wallStart).count();
  if (!std::isfinite(seconds) || seconds <= 0.0)
    throw std::runtime_error{"Invalid timing event interval"};
  return {seconds, wall};
}

void validateAnalytic(auto const &queue,
                      alpaka::concepts::IBuffer auto const &result,
                      alpaka::concepts::IBuffer auto &host, Parameters p,
                      Index steps) {
  alpaka::onHost::memcpy(queue, host, result);
  alpaka::onHost::wait(queue);
  auto const [valid, error] =
      ::validateSolution(host.getMdSpan(), Vec{p.size + 2u, p.size + 2u}, p.dx,
                         p.dy, steps * p.dt);
  if (!valid)
    throw std::runtime_error{"Analytic validation failed, max error " +
                             std::to_string(error)};
}

auto choices(Index size, alpaka::onHost::DeviceProperties const &properties,
             bool sequential = false) -> std::vector<Choice> {
  std::vector<Choice> result;
  auto const threadLimits = properties.getMaxThreadsPerBlock<2u>();
  auto const blockLimits = properties.getMaxBlocksPerGrid<2u>();
  for (Index variant = 0; variant < layouts.size(); ++variant) {
    auto const layout = layouts[variant];
    if ((!sequential && (layout.workers() > properties.maxThreadsPerBlock ||
                         layout.threadsX > threadLimits.x() ||
                         layout.threadsY > threadLimits.y())) ||
        (properties.sharedMemPerBlockBytes &&
         layout.sharedBytes() > properties.sharedMemPerBlockBytes))
      continue;
    if (variant == 0u) {
      if (size % 16u == 0u)
        result.push_back({0u, 0u});
      continue;
    }
    for (Index policy = 0; policy < 3u; ++policy) {
      Choice const choice{variant, policy};
      auto const count =
          gridBlocks(choice, size, properties.multiProcessorCount);
      if (count > blockLimits.x() || count > properties.maxBlocksPerGrid)
        continue;
      auto duplicate = false;
      for (auto const &other : result)
        if (other.variant == variant &&
            gridBlocks(other, size, properties.multiProcessorCount) == count)
          duplicate = true;
      if (!duplicate)
        result.push_back(choice);
    }
  }
  return result;
}

#if !ALPAKA_TUNE_HEAT_REPLAY
void selfTest(alpaka::concepts::BackendSpec auto backend,
              Options const &options) {
  auto const executor = alpaka::getExecutor(backend);
  auto device =
      alpaka::onHost::makeDeviceSelector(alpaka::onHost::DeviceSpec{backend})
          .makeDevice(options.device);
  auto queue =
      device.makeQueue(alpaka::queueKind::nonBlocking, alpaka::timing::enabled);
  auto const properties = device.getDeviceProperties();
  for (auto size : options.sizes) {
    Parameters const p = parameters(size);
    auto initial = alpaka::onHost::allocHost<double>(Vec{size + 2u, size + 2u});
    auto reference =
        alpaka::onHost::allocHost<double>(Vec{size + 2u, size + 2u});
    auto referenceNext =
        alpaka::onHost::allocHost<double>(Vec{size + 2u, size + 2u});
    auto actual = alpaka::onHost::allocHost<double>(Vec{size + 2u, size + 2u});
    ::initalizeBuffer(initial.getMdSpan(), p.dx, p.dy);
    std::mt19937_64 random{options.seed};
    for (Index y = 1; y <= size; ++y)
      for (Index x = 1; x <= size; ++x)
        initial[Vec{y, x}] = double(random() % 100000u) / 100000.0;
    for (Index y = 0; y < size + 2u; ++y)
      for (Index x = 0; x < size + 2u; ++x)
        reference[Vec{y, x}] = initial[Vec{y, x}];
    // Independent host recurrence, including the same analytic boundaries.
    for (Index step = 1; step <= 7u; ++step) {
      double const rX = p.dt / (p.dx * p.dx);
      double const rY = p.dt / (p.dy * p.dy);
      for (Index y = 0; y < size + 2u; ++y)
        for (Index x = 0; x < size + 2u; ++x)
          if (x == 0u || y == 0u || x == size + 1u || y == size + 1u)
            referenceNext[Vec{y, x}] =
                ::exactSolution(x * p.dx, y * p.dy, step * p.dt);
          else
            referenceNext[Vec{y, x}] =
                reference[Vec{y, x}] * (1.0 - 2.0 * rX - 2.0 * rY) +
                reference[Vec{y, x - 1u}] * rX +
                reference[Vec{y, x + 1u}] * rX +
                reference[Vec{y - 1u, x}] * rY + reference[Vec{y + 1u, x}] * rY;
      for (Index y = 0; y < size + 2u; ++y)
        for (Index x = 0; x < size + 2u; ++x)
          reference[Vec{y, x}] = referenceNext[Vec{y, x}];
    }
    auto current = alpaka::onHost::allocLike(device, initial);
    auto next = alpaka::onHost::allocLike(device, initial);
    auto const legal =
        choices(size, properties,
                alpaka::exec::isSeqExecutor_v<ALPAKA_TYPEOF(executor)>);
    for (auto choice : legal) {
      alpaka::onHost::memcpy(queue, current, initial);
      alpaka::onHost::memcpy(queue, next, initial);
      dispatch(choice.variant, [&](auto variant) {
        for (Index step = 1; step <= 7u; ++step) {
          launch(variant, choice, properties.multiProcessorCount, queue,
                 executor, current, next, p);
          boundary(queue, executor, next, p, step);
          std::swap(current, next);
        }
      });
      alpaka::onHost::memcpy(queue, actual, current);
      alpaka::onHost::wait(queue);
      for (Index y = 0; y < size + 2u; ++y)
        for (Index x = 0; x < size + 2u; ++x) {
          double const expected = reference[Vec{y, x}];
          double const found = actual[Vec{y, x}];
          if (!std::isfinite(found) ||
              std::abs(found - expected) >
                  64.0 * std::numeric_limits<double>::epsilon() *
                      (1.0 + std::abs(expected)))
            throw std::runtime_error{
                "Recurrence mismatch: size=" + std::to_string(size) +
                " variant=" + std::to_string(choice.variant) +
                " policy=" + std::to_string(choice.gridPolicy) + " at " +
                std::to_string(y) + "," + std::to_string(x)};
        }
    }
    std::cout
        << "Correctness " << size << 'x' << size << ": " << legal.size()
        << " legal variant/grid pairs, 7 steps, all cells and halos passed\n";
  }
}

auto tune(alpaka::onHost::concepts::Device auto &device, auto const &queue,
          alpaka::concepts::Executor auto executor,
          alpaka::concepts::IBuffer auto const &input,
          alpaka::concepts::IBuffer auto &output, Parameters p,
          Options const &options, Metadata const &meta, Output &files,
          double &tuningSeconds, Index &legalCount) -> Choice {
  auto const properties = device.getDeviceProperties();
  auto const legal =
      choices(p.size, properties,
              alpaka::exec::isSeqExecutor_v<ALPAKA_TYPEOF(executor)>);
  legalCount = static_cast<Index>(legal.size());
  if (legal.empty())
    throw std::runtime_error{"No legal stencil candidates"};
  auto const baseline =
      alpakaTune::deriveThreadSpec(device, originalFrame(p.size, executor),
                                   originalBundle(input, output, p));
  std::vector<Vec> blockChoices, threadChoices;
  auto append = [](auto &values, auto value) {
    if (std::ranges::find(values, value) == values.end())
      values.push_back(value);
  };
  auto blocksFor = [&](Choice choice) {
    return choice.variant == 0u
               ? Vec{baseline.getNumBlocks()}
               : Vec{1u, gridBlocks(choice, p.size,
                                    properties.multiProcessorCount)};
  };
  auto threadsFor = [&](Choice choice) {
    auto const layout = layouts[choice.variant];
    return choice.variant == 0u
               ? Vec{baseline.getNumThreads()}
               : (alpaka::exec::isSeqExecutor_v<ALPAKA_TYPEOF(executor)>
                      ? Vec{1u, 1u}
                      : Vec{layout.threadsY, layout.threadsX});
  };
  for (auto choice : legal) {
    append(blockChoices, blocksFor(choice));
    append(threadChoices, threadsFor(choice));
  }
  auto tunables = alpakaTune::constrain(
      alpakaTune::TunableBundle{
          variantName(alpakaTune::CValsTo<lastVariant>{}),
          alpakaTune::numBlocks(alpakaTune::RVals<Vec>{blockChoices},
                                alpakaTune::mdPolicy::listed),
          alpakaTune::numThreads(alpakaTune::RVals<Vec>{threadChoices},
                                 alpakaTune::mdPolicy::listed)},
      alpakaTune::restrict(
          variantName, alpakaTune::numBlocks,
          [legal, blocksFor](Index variant, Vec const &blocks) {
            return std::ranges::any_of(legal, [&](Choice choice) {
              return choice.variant == variant && blocks == blocksFor(choice);
            });
          }),
      alpakaTune::restrict(
          variantName, alpakaTune::numThreads,
          [legal, threadsFor](Index variant, Vec const &threads) {
            return std::ranges::any_of(legal, [&](Choice choice) {
              return choice.variant == variant && threads == threadsFor(choice);
            });
          }));
  auto config = alpakaTune::TunerConfig{};
  config.strategy = alpakaTune::StrategyKind::exhaustive;
  config.selection = alpakaTune::SelectionPolicy::fixed;
  config.runsPerCandidate = config.minimumRunsPerCandidate = options.samples;
  config.historyWindowSize = options.samples;
  config.mannWhitneyEarlyStop = false;
  config.randomSeed = options.seed;
  config.maximumConsecutiveStrategyRetries =
      layouts.size() * blockChoices.size() * threadChoices.size() + 1u;
  config.maximumExecutions = legal.size() * (options.samples + 3u) * 4u;
  config.maximumRetiredConfigurations.reset();
  config.queue =
      alpakaTune::QueueConfig{.warmupRuns = 2u,
                              .noiseCancellationWindow = legal.size(),
                              .maxConsecutiveRuns = 3u};
  config.history = {.read = false, .write = false};
  config.completeHistory = {.read = false, .write = false};
  auto tuner = alpakaTune::makeTuner(
      config, tunables, device, executor,
      alpakaTune::elapsedTimeMetric("batched_stencil_seconds"),
      "heatEquation/v" + std::to_string(catalogVersion) + "/fp64/" +
          std::to_string(p.size));
  auto const prototype =
      alpaka::KernelBundle{TunedStencil{}, input, output, p, variantName};
  auto const launchSpec = alpaka::onHost::ThreadSpec{
      Vec{baseline.getNumBlocks()}, Vec{baseline.getNumThreads()}, executor};
  auto decode = [&](auto const &normalized) {
    auto const variant = static_cast<Index>(
        std::lround(normalized.at(0u) * double(lastVariant)));
    auto const blockIndex = static_cast<std::size_t>(
        std::lround(normalized.at(1u) * double(blockChoices.size() - 1u)));
    if (blockIndex >= blockChoices.size())
      throw std::runtime_error{"Invalid normalized block choice"};
    for (auto choice : legal)
      if (choice.variant == variant &&
          blocksFor(choice) == blockChoices[blockIndex])
        return choice;
    throw std::runtime_error{"Tuner selected an uncoupled launch"};
  };
  auto baselineLaunch = [&] {
    launch(std::integral_constant<Index, 0u>{}, Choice{0u, 0u}, meta.sms, queue,
           executor, input, output, p);
  };
  for (Index warm = 0; warm < 5u; ++warm)
    baselineLaunch();
  auto const calibration = measure(device, queue,
                                   [&] {
                                     for (Index run = 0; run < 16u; ++run)
                                       baselineLaunch();
                                   })
                               .queue /
                           16.0;
  Index const batch = static_cast<Index>(
      std::clamp(std::ceil(0.010 / calibration), 1.0, 4096.0));
  auto const start = Clock::now();
  std::vector<std::vector<double>> measured(legal.size());
  std::size_t launches{};
  while (!tuner.completed()) {
    if (++launches > *config.maximumExecutions)
      throw std::runtime_error{"Bounded tuner launch count exceeded"};
    tuner.enqueue(queue, launchSpec, prototype);
    auto const choice = decode(tuner.lastConfig().configuration);
    auto const seconds = dispatch(choice.variant, [&](auto variant) {
      return measure(device, queue,
                     [&] {
                       for (Index run = 0; run < batch; ++run)
                         launch(variant, choice, meta.sms, queue, executor,
                                input, output, p);
                     })
                 .queue /
             batch;
    });
    tuner.provideMetric(seconds);
    auto const &record = tuner.lastConfig();
    if (record.measured) {
      auto const position =
          std::ranges::find_if(legal,
                               [&](Choice other) {
                                 return choice.variant == other.variant &&
                                        choice.gridPolicy == other.gridPolicy;
                               }) -
          legal.begin();
      measured.at(position).push_back(seconds);
    }
    if (files.tuning)
      files.tuning << prefix(meta, p.size) << launches << ',' << choice.variant
                   << ',' << choice.gridPolicy << ',' << record.measured << ','
                   << batch << ',' << seconds << '\n';
  }
  if (tuner.info().measuredCandidateCount != legal.size())
    throw std::runtime_error{
        "Exhaustive tuning did not cover every legal candidate"};
  std::vector<std::size_t> order(legal.size());
  std::iota(order.begin(), order.end(), 0u);
  std::ranges::sort(order, [&](auto a, auto b) {
    return median(measured[a]) < median(measured[b]);
  });
  order.resize(std::min<std::size_t>(5u, order.size()));
  auto const defaultRow =
      std::ranges::find_if(legal, [](Choice c) { return c.variant == 0u; }) -
      legal.begin();
  if (std::ranges::find(order, defaultRow) == order.end())
    order.push_back(defaultRow);
  std::vector<std::vector<double>> finalistSamples(order.size());
  std::vector<std::size_t> rotations(order.size());
  std::iota(rotations.begin(), rotations.end(), 0u);
  std::mt19937_64 random{options.seed + p.size};
  for (Index repetition = 0; repetition < std::max(31u, options.repetitions);
       ++repetition) {
    std::shuffle(rotations.begin(), rotations.end(), random);
    for (auto position : rotations) {
      auto choice = legal[order[position]];
      auto seconds = dispatch(choice.variant, [&](auto variant) {
        return measure(device, queue,
                       [&] {
                         for (Index run = 0; run < batch; ++run)
                           launch(variant, choice, meta.sms, queue, executor,
                                  input, output, p);
                       })
                   .queue /
               batch;
      });
      finalistSamples[position].push_back(seconds);
      if (files.finalists)
        files.finalists << prefix(meta, p.size) << choice.variant << ','
                        << choice.gridPolicy << ',' << repetition << ','
                        << batch << ',' << seconds << '\n';
    }
  }
  std::size_t best{};
  for (std::size_t position = 1; position < order.size(); ++position)
    if (median(finalistSamples[position]) < median(finalistSamples[best]))
      best = position;
  tuningSeconds = std::chrono::duration<double>(Clock::now() - start).count();
  std::cout << "Tuned " << p.size << ": " << legal.size()
            << " legal candidates, " << launches << " tuner launches, batch "
            << batch << ", winner " << legal[order[best]].variant << '/'
            << legal[order[best]].gridPolicy << ", " << tuningSeconds << " s\n";
  return legal[order[best]];
}
#endif

template <Index Variant>
void evaluate(std::integral_constant<Index, Variant> variant, Choice choice,
              alpaka::onHost::concepts::Device auto &device, auto const &queue,
              alpaka::concepts::Executor auto executor,
              alpaka::concepts::IBuffer auto const &initial,
              alpaka::concepts::IBuffer auto &host,
              alpaka::concepts::IBuffer auto &current,
              alpaka::concepts::IBuffer auto &next, Parameters p,
              Options const &options, Metadata const &meta, Output &files,
              Index legalCount, double tuningSeconds) {
  auto reset = [&] {
    alpaka::onHost::memcpy(queue, current, initial);
    alpaka::onHost::memcpy(queue, next, initial);
    alpaka::onHost::wait(queue);
  };
  auto stencil = [&](bool tuned) {
    if (tuned)
      launch(variant, choice, meta.sms, queue, executor, current, next, p);
    else
      launch(std::integral_constant<Index, 0u>{}, Choice{0, 0}, meta.sms, queue,
             executor, current, next, p);
  };
  auto simulation = [&](bool tuned, Index steps) {
    for (Index step = 1; step <= steps; ++step) {
      stencil(tuned);
      boundary(queue, executor, next, p, step);
      std::swap(current, next);
    }
  };
  if (!options.profile.empty()) {
    reset();
    simulation(options.profile == "winner", options.steps);
    alpaka::onHost::wait(queue);
    validateAnalytic(queue, current, host, p, options.steps);
    std::cout << "Profile workload " << options.profile << ' ' << p.size
              << ": correctness passed\n";
    return;
  }
  std::array<std::vector<double>, 2> stepTimes, stencilTimes, wallTimes;
  std::string const mode = ALPAKA_TUNE_HEAT_REPLAY ? "replay" : "search";
  for (Index repetition = 0; repetition < options.repetitions + 3u;
       ++repetition)
    for (Index offset = 0; offset < 2u; ++offset) {
      Index const implementation = (repetition + offset) % 2u;
      reset();
      auto const duration = measure(device, queue, [&] {
        simulation(implementation != 0u, options.steps);
      });
      if (repetition == 0u || repetition + 1u == options.repetitions + 3u)
        validateAnalytic(queue, current, host, p, options.steps);
      reset();
      auto const isolated = measure(device, queue, [&] {
        for (Index run = 0; run < options.steps; ++run)
          stencil(implementation != 0u);
      });
      if (repetition < 3u)
        continue;
      auto const stepSeconds = duration.queue / options.steps;
      auto const stencilSeconds = isolated.queue / options.steps;
      stepTimes[implementation].push_back(stepSeconds);
      stencilTimes[implementation].push_back(stencilSeconds);
      wallTimes[implementation].push_back(duration.wall / options.steps);
      if (files.samples) {
        auto const label = implementation ? "winner" : "default";
        files.samples << prefix(meta, p.size) << mode << ',' << label << ','
                      << repetition - 3u << ",complete_step," << stepSeconds
                      << ',' << duration.wall / options.steps << ','
                      << options.steps << '\n';
        files.samples << prefix(meta, p.size) << mode << ',' << label << ','
                      << repetition - 3u << ",stencil," << stencilSeconds << ','
                      << isolated.wall / options.steps << ',' << options.steps
                      << '\n';
      }
    }
  auto const baseline = median(stepTimes[0]);
  auto const optimized = median(stepTimes[1]);
  auto const confidence =
      options.repetitions >= 5u
          ? interval(stepTimes[0], stepTimes[1], options.seed + p.size)
          : std::array<double, 2>{0, 0};
  std::cout << p.size << 'x' << p.size << ": default " << baseline * 1e3
            << " ms/step; winner " << optimized * 1e3 << " ms/step; "
            << baseline / optimized << "x [" << confidence[0] << ','
            << confidence[1] << "]; variant " << choice.variant << '/'
            << choice.gridPolicy << '\n';
  if (files.summary) {
    auto const layout = layouts[choice.variant];
    files.summary << prefix(meta, p.size) << mode << ",fp64," << catalogVersion
                  << ',' << legalCount << ',' << choice.variant << ','
                  << choice.gridPolicy << ','
                  << static_cast<Index>(layout.implementation) << ','
                  << layout.threadsX << ',' << layout.threadsY << ','
                  << layout.rows << ',' << layout.padding << ','
                  << gridBlocks(choice, p.size, meta.sms) << ','
                  << options.steps << ',' << tuningSeconds << ',' << baseline
                  << ',' << optimized << ',' << median(stencilTimes[0]) << ','
                  << median(stencilTimes[1]) << ',' << baseline / optimized
                  << ',' << confidence[0] << ',' << confidence[1] << ','
                  << median(wallTimes[0]) << ',' << median(wallTimes[1])
                  << '\n';
    files.summary.flush();
  }
}

void run(alpaka::concepts::BackendSpec auto backend, Options const &options,
         Output &files) {
  auto const executor = alpaka::getExecutor(backend);
  auto device =
      alpaka::onHost::makeDeviceSelector(alpaka::onHost::DeviceSpec{backend})
          .makeDevice(options.device);
  auto queue =
      device.makeQueue(alpaka::queueKind::nonBlocking, alpaka::timing::enabled);
  auto const meta = metadata(device, executor);
  std::cout << meta.device << " / " << meta.architecture << " / "
            << meta.backend << " / " << meta.executor << '\n';
#if ALPAKA_TUNE_HEAT_REPLAY
  if (selectedDevice.empty() || meta.device != selectedDevice ||
      meta.architecture != selectedArchitecture ||
      meta.executor != selectedExecutor || catalogVersion != selectedCatalog ||
      meta.sms != selectedMultiProcessors ||
      meta.ompThreads != selectedOmpThreads)
    throw std::runtime_error{
        "Inserted winners do not match the device, architecture, executor, "
        "catalog, or thread configuration"};
#else
  if (options.selfTest) {
    selfTest(backend, options);
    return;
  }
  if (!options.profile.empty())
    throw std::invalid_argument{"Profile the inserted replay executable"};
  if constexpr (alpaka::exec::isSeqExecutor_v<ALPAKA_TYPEOF(executor)> &&
                !std::same_as<ALPAKA_TYPEOF(executor),
                              alpaka::exec::CpuOmpBlocks>)
    throw std::invalid_argument{
        "Use --self-test for serial execution; "
        "performance tuning requires a GPU or ompBlocks"};
#endif
  std::ofstream winners;
  if (options.exportWinners) {
    winners.open(*options.exportWinners);
    if (!winners)
      throw std::runtime_error{"Cannot export winner header"};
    winners << "// Copyright 2026 Tim Hanel\n// SPDX-License-Identifier: "
               "MPL-2.0\n#pragma once\n"
               "#include <array>\n#include <cstdint>\n#include <string_view>\n"
               "namespace alpakaTune::benchmarks::heatEquation {\n"
               "struct SelectedWinner { std::uint32_t size, variant, "
               "gridPolicy; };\n"
            << "inline constexpr std::string_view selectedDevice = "
            << cppString(meta.device) << ";\n"
            << "inline constexpr std::string_view selectedArchitecture = "
            << cppString(meta.architecture) << ";\n"
            << "inline constexpr std::string_view selectedExecutor = "
            << cppString(meta.executor) << ";\n"
            << "inline constexpr std::uint32_t selectedCatalog = "
            << catalogVersion << ", selectedMultiProcessors = " << meta.sms
            << ", selectedOmpThreads = " << meta.ompThreads << ";\n"
            << "inline constexpr std::array selectedWinners{\n";
  }
  if (options.csv) {
    std::ofstream provenance{options.csv->string() + ".metadata.json"};
    provenance << "{\n\"device\":" << quote(meta.device)
               << ",\n\"architecture\":" << quote(meta.architecture)
               << ",\n\"backend\":" << quote(meta.backend)
               << ",\n\"executor\":" << quote(meta.executor)
               << ",\n\"gpu_uuid\":" << quote(meta.uuid)
               << ",\n\"sm_count\":" << meta.sms
               << ",\n\"omp_threads\":" << meta.ompThreads
               << ",\n\"cuda_driver_version\":" << meta.driver
               << ",\n\"cuda_runtime_version\":" << meta.runtime
               << ",\n\"source_revision\":\"" ALPAKA_TUNE_HEAT_SOURCE_REVISION
                  "\",\n"
                  "\"catalog\":"
               << catalogVersion
               << ",\n\"precision\":\"fp64\",\n\"seed\":" << options.seed
               << ",\n\"samples_per_candidate\":" << options.samples
               << ",\n\"repetitions\":" << options.repetitions
               << ",\n\"steps\":" << options.steps << "\n}\n";
  }
  for (auto size : options.sizes) {
    Parameters const p = parameters(size);
    auto initial = alpaka::onHost::allocHost<double>(Vec{size + 2u, size + 2u});
    auto host = alpaka::onHost::allocHost<double>(Vec{size + 2u, size + 2u});
    ::initalizeBuffer(initial.getMdSpan(), p.dx, p.dy);
    auto current = alpaka::onHost::allocLike(device, initial);
    auto next = alpaka::onHost::allocLike(device, initial);
    auto savedInitial = alpaka::onHost::allocLike(device, initial);
    alpaka::onHost::memcpy(queue, savedInitial, initial);
    alpaka::onHost::memcpy(queue, current, initial);
    alpaka::onHost::memcpy(queue, next, initial);
    alpaka::onHost::wait(queue);
    Index legalCount{};
    double tuningSeconds{};
#if ALPAKA_TUNE_HEAT_REPLAY
    dispatchSelected(size, [&](auto variant, Choice choice) {
      evaluate(variant, choice, device, queue, executor, savedInitial, host,
               current, next, p, options, meta, files, legalCount,
               tuningSeconds);
    });
#else
    auto const choice = tune(device, queue, executor, current, next, p, options,
                             meta, files, tuningSeconds, legalCount);
    dispatch(choice.variant, [&](auto variant) {
      evaluate(variant, choice, device, queue, executor, savedInitial, host,
               current, next, p, options, meta, files, legalCount,
               tuningSeconds);
    });
    if (winners)
      winners << "SelectedWinner{" << size << ',' << choice.variant << ','
              << choice.gridPolicy << "},\n";
#endif
    alpaka::onHost::wait(queue);
  }
  if (winners)
    winners << "};\n} // namespace alpakaTune::benchmarks::heatEquation\n";
}
} // namespace alpakaTune::benchmarks::heatEquation

int main(int argc, char **argv) {
  using namespace alpakaTune::benchmarks::heatEquation;
  try {
    if (!alpakaTune::consumeBackendOptions(argc, argv))
      return EXIT_FAILURE;
    auto const options = parseOptions(argc, argv);
    Output output{options};
    bool ran{};
    alpaka::onHost::executeForEach(
        [&](alpaka::concepts::BackendSpec auto backend) {
          if (alpakaTune::backendSelected(backend)) {
            auto selector = alpaka::onHost::makeDeviceSelector(
                alpaka::onHost::DeviceSpec{backend});
            if (!selector.isAvailable())
              return EXIT_SUCCESS;
            run(backend, options, output);
            ran = true;
          }
          return EXIT_SUCCESS;
        },
        alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs,
                                    alpaka::exec::enabledExecutors));
    if (!ran)
      throw std::runtime_error{"No selected device/executor available"};
    return EXIT_SUCCESS;
  } catch (std::exception const &error) {
    std::cerr << "Heat equation: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
