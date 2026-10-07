// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#include "MatmulKernel.hpp"

#include <alpakaTune/BackendSelection.hpp>
#include <alpakaTune/alpakaTune.hpp>

#if ALPAKA_TUNE_GEMM_VENDOR
#include <alpaka/blas.hpp>
#endif

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace alpakaTune::benchmarks::matmul {
inline constexpr auto variantName = ALPAKA_TUNE_TUNABLE("tileVariant");
using Clock = std::chrono::steady_clock;

struct Options {
  std::vector<Shape> shapes;
  Index device{};
  Index repetitions{15};
  Index samplesPerCandidate{7};
  bool selfTest{};
  bool kernelOnly{};
  bool reuseHistory{};
  std::optional<std::filesystem::path> csv;
  std::optional<std::filesystem::path> history;
};

auto positive(std::string_view text) -> Index {
  Index value{};
  auto const [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size() || value == 0u)
    throw std::invalid_argument{"Expected a positive 32-bit integer: " +
                                std::string{text}};
  return value;
}

auto parseShape(std::string_view text) -> Shape {
  auto const first = text.find('x');
  auto const second = text.find('x', first == text.npos ? 0u : first + 1u);
  if (first == text.npos || second == text.npos)
    throw std::invalid_argument{"--shape expects MxNxK"};
  auto shape = Shape{positive(text.substr(0u, first)),
                     positive(text.substr(first + 1u, second - first - 1u)),
                     positive(text.substr(second + 1u))};
  // The kernel uses 32-bit coordinates but pitches and allocation sizes use
  // size_t. Leave headroom for the last tile and vendor's signed descriptors.
  constexpr Index maximumDimension = 1u << 20u;
  if (std::max({shape.m, shape.n, shape.k}) > maximumDimension)
    throw std::invalid_argument{"Matrix dimensions must not exceed 1048576"};
  return shape;
}

auto parseOptions(int argc, char **argv) -> Options {
  auto options = Options{};
  for (int argument = 1; argument < argc; ++argument) {
    auto const name = std::string_view{argv[argument]};
    auto value = [&]() -> std::string_view {
      if (++argument == argc)
        throw std::invalid_argument{std::string{name} + " requires a value"};
      return argv[argument];
    };
    if (name == "--shape")
      options.shapes.push_back(parseShape(value()));
    else if (name == "--repetitions")
      options.repetitions = positive(value());
    else if (name == "--samples-per-candidate")
      options.samplesPerCandidate = positive(value());
    else if (name == "--device") {
      auto const text = value();
      options.device = text == "0" ? 0u : positive(text);
    } else if (name == "--csv")
      options.csv = std::filesystem::path{value()};
    else if (name == "--history")
      options.history = std::filesystem::path{value()};
    else if (name == "--reuse-history")
      options.reuseHistory = true;
    else if (name == "--self-test")
      options.selfTest = true;
    else if (name == "--kernel-only")
      options.kernelOnly = true;
    else if (name == "--help") {
      std::cout
          << "C++ FP32 GEMM: default kernel, tuned kernel, alpakaVendor\n"
             "  --backend api:deviceKind --executor name --device INDEX\n"
             "  --shape MxNxK (repeatable) --repetitions N\n"
             "  --samples-per-candidate N --csv FILE\n"
             "  --history FILE --reuse-history --kernel-only --self-test\n";
      std::exit(EXIT_SUCCESS);
    } else
      throw std::invalid_argument{"Unknown argument: " + std::string{name}};
  }
  if (options.reuseHistory && !options.history)
    throw std::invalid_argument{"--reuse-history requires --history"};
#if !ALPAKA_TUNE_HAS_JSON
  if (options.history)
    throw std::invalid_argument{
        "--history requires alpakaTune_ENABLE_PERSISTENCE=ON"};
#endif
  if (options.samplesPerCandidate > 1000u || options.repetitions > 10000u)
    throw std::invalid_argument{
        "Too many samples (maximum 1000 tuning / 10000 evaluation)"};
  if (options.selfTest && options.reuseHistory)
    throw std::invalid_argument{"--self-test requires fresh tuning"};
  if (options.shapes.empty())
    options.shapes = options.selfTest ? std::vector<Shape>{{1, 1, 1},
                                                           {17, 29, 19},
                                                           {64, 64, 32},
                                                           {65, 67, 33}}
                                      : std::vector<Shape>{{512, 512, 512},
                                                           {1024, 1024, 1024},
                                                           {2048, 2048, 2048},
                                                           {2048, 256, 1024},
                                                           {256, 2048, 1024}};
  return options;
}

auto quote(std::string_view value) -> std::string {
  auto result = std::string{"\""};
  for (char character : value) {
    if (character == '"')
      result += '"';
    result += character;
  }
  return result + '"';
}

struct Output {
  std::ofstream summary, samples, tuning;
  explicit Output(Options const &options) {
    if (!options.csv)
      return;
    auto const path = *options.csv;
    if (options.history &&
        std::filesystem::absolute(path).lexically_normal() ==
            std::filesystem::absolute(*options.history).lexically_normal())
      throw std::invalid_argument{"CSV and history must use different paths"};
    if (!path.parent_path().empty())
      std::filesystem::create_directories(path.parent_path());
    summary.open(path);
    samples.open(path.string() + ".samples.csv");
    tuning.open(path.string() + ".tuning.csv");
    if (!summary || !samples || !tuning)
      throw std::runtime_error{"Cannot open CSV output files"};
    summary
        << "device,backend,executor,m,n,k,precision,strategy,legal_variants,"
           "winner,"
           "block_m,block_n,block_k,thread_m,thread_n,stages,history_reused,"
           "tuning_launches,tuning_wall_s,baseline_median_s,tuned_median_s,"
           "vendor_median_s,vendor,improvement_percent,speedup_vs_default,"
           "speedup_vs_vendor,break_even_launches,baseline_wall_median_s,"
           "tuned_wall_median_s,vendor_wall_median_s,wall_break_even_"
           "launches,default_speedup_ci_low,default_speedup_ci_high,"
           "vendor_speedup_ci_low,vendor_speedup_ci_high\n";
    samples << "device,backend,executor,m,n,k,implementation,repetition,queue_"
               "s,wall_s\n";
    tuning << "device,backend,executor,m,n,k,candidate,execution,measured,"
              "queue_s\n";
    summary << std::setprecision(12);
    samples << std::setprecision(12);
    tuning << std::setprecision(12);
  }
};

struct ReferencePoint {
  Index row, column;
  double value, tolerance;
};

auto reference(alpaka::concepts::IBuffer auto const &a,
               alpaka::concepts::IBuffer auto const &b, Shape shape, bool full)
    -> std::vector<ReferencePoint> {
  auto points = std::vector<ReferencePoint>{};
  auto add = [&](Index row, Index column) {
    double expected{}, magnitude{};
    for (Index inner = 0; inner < shape.k; ++inner) {
      auto const term = double{a[alpaka::Vec{row, inner}]} *
                        double{b[alpaka::Vec{inner, column}]};
      expected += term;
      magnitude += std::abs(term);
    }
    // Forward-error bound for an FP32 dot product, plus a small absolute floor.
    auto const epsilon = double{std::numeric_limits<float>::epsilon()};
    auto const gamma = shape.k * epsilon / (1.0 - shape.k * epsilon);
    points.push_back({row, column, expected, 3.0 * gamma * magnitude + 2.0e-6});
  };
  if (full) {
    for (Index row = 0; row < shape.m; ++row)
      for (Index column = 0; column < shape.n; ++column)
        add(row, column);
  } else {
    for (auto row : {0u, shape.m / 2u, shape.m - 1u})
      for (auto column : {0u, shape.n / 2u, shape.n - 1u})
        add(row, column);
    auto generator = std::mt19937{42u};
    for (Index sample = 0; sample < 64; ++sample)
      add(generator() % shape.m, generator() % shape.n);
  }
  return points;
}

void check(alpaka::concepts::IBuffer auto const &c,
           std::vector<ReferencePoint> const &points,
           std::string const &label) {
  for (auto point : points) {
    auto const actual = double{c[alpaka::Vec{point.row, point.column}]};
    if (!std::isfinite(actual) ||
        std::abs(actual - point.value) > point.tolerance)
      throw std::runtime_error{
          label + " failed at C[" + std::to_string(point.row) + "," +
          std::to_string(point.column) + "]: actual=" + std::to_string(actual) +
          ", expected=" + std::to_string(point.value)};
  }
}

// Runtime dispatch instantiates precisely the same 12 variants used by CVals.
template <Index Variant = 0u>
void launch(Index variant, alpaka::onHost::concepts::Device auto const &device,
            auto const &queue, alpaka::concepts::Executor auto executor,
            alpaka::concepts::IBuffer auto const &a,
            alpaka::concepts::IBuffer auto const &b,
            alpaka::concepts::IBuffer auto &c, Shape shape, bool vectorLoads,
            Index threads) {
  if (variant == Variant) {
    queue.enqueue(
        alpaka::onHost::ThreadSpec{tiles[Variant].blocks(shape), threads,
                                   executor},
        alpaka::KernelBundle{MatmulKernel{}, a, b, c, shape, vectorLoads,
                             std::integral_constant<Index, Variant>{}});
  } else if constexpr (Variant + 1u < tiles.size()) {
    launch<Variant + 1u>(variant, device, queue, executor, a, b, c, shape,
                         vectorLoads, threads);
  } else
    throw std::out_of_range{"Unknown GEMM tile variant"};
}

auto median(std::vector<double> values) -> double {
  std::sort(values.begin(), values.end());
  auto const middle = values.size() / 2u;
  return values.size() % 2u ? values[middle]
                            : (values[middle - 1u] + values[middle]) / 2.0;
}

// Paired resampling preserves the common device state of each evaluation
// repetition. Evaluation samples are independent of the tuning measurements.
auto speedupInterval(std::vector<double> const &baseline,
                     std::vector<double> const &optimized)
    -> std::optional<std::array<double, 2>> {
  if (baseline.size() < 5u)
    return std::nullopt;
  auto random = std::mt19937{712u};
  auto select =
      std::uniform_int_distribution<std::size_t>{0u, baseline.size() - 1u};
  auto speedups = std::vector<double>{};
  auto a = std::vector<double>(baseline.size());
  auto b = std::vector<double>(baseline.size());
  constexpr std::size_t resamples = 1000u;
  for (std::size_t sample = 0u; sample < resamples; ++sample) {
    for (std::size_t repetition = 0u; repetition < a.size(); ++repetition) {
      auto const index = select(random);
      a[repetition] = baseline[index];
      b[repetition] = optimized[index];
    }
    speedups.push_back(median(a) / median(b));
  }
  std::sort(speedups.begin(), speedups.end());
  return std::array{speedups[25u], speedups[974u]};
}

auto breakEven(double cost, double baseline, double optimized)
    -> std::optional<double> {
  if (optimized >= baseline)
    return std::nullopt;
  return std::ceil(cost / (baseline - optimized));
}

template <alpaka::onHost::concepts::Device Device>
constexpr auto vendorName() -> std::string_view {
#if ALPAKA_TUNE_GEMM_VENDOR
  using Api = ALPAKA_TYPEOF(std::declval<Device>().getApi());
  if constexpr (std::same_as<Api, alpaka::api::Cuda> && ALPAKAV_DEP_CUBLAS)
    return "cuBLAS via alpakaVendor";
  else if constexpr (std::same_as<Api, alpaka::api::Hip> && ALPAKAV_DEP_ROCBLAS)
    return "rocBLAS via alpakaVendor";
  else if constexpr (std::same_as<Api, alpaka::api::OneApi> &&
                     ALPAKAV_DEP_ONEMKL)
    return "oneMKL via alpakaVendor";
  else if constexpr (std::same_as<Api, alpaka::api::Host> &&
                     ALPAKAV_DEP_OPENBLAS)
    return "OpenBLAS via alpakaVendor";
#endif
  return {};
}

void run(alpaka::concepts::BackendSpec auto backend, Options const &options,
         Output &output) {
  auto const executor = alpaka::getExecutor(backend);
  auto const spec = alpaka::onHost::DeviceSpec{backend};
  auto selector = alpaka::onHost::makeDeviceSelector(spec);
  auto device = selector.makeDevice(options.device);
  auto queue =
      device.makeQueue(alpaka::queueKind::nonBlocking, alpaka::timing::enabled);
  auto const properties = device.getDeviceProperties();
  auto const threads =
      alpaka::exec::isSeqExecutor_v<ALPAKA_TYPEOF(executor)> ? 1u : 256u;
  auto const vendor = vendorName<ALPAKA_TYPEOF(device)>();
  if (vendor.empty() && !options.kernelOnly && !options.selfTest)
    throw std::runtime_error{"No vendor BLAS enabled for this backend; enable "
                             "its alpakaV_DEP option or use --kernel-only"};
  auto const useVendor = !vendor.empty() && !options.kernelOnly;
  std::cout << device.getName() << " / " << spec.getApi().getName() << " / "
            << alpaka::onHost::demangledName(executor) << '\n';

  for (auto shape : options.shapes) {
    using Matrix = alpaka::Vec<std::size_t, 2u>;
    auto aHost = alpaka::onHost::allocHost<float>(Matrix{shape.m, shape.k});
    auto bHost = alpaka::onHost::allocHost<float>(Matrix{shape.k, shape.n});
    auto cHost = alpaka::onHost::allocHost<float>(Matrix{shape.m, shape.n});
    auto random = std::mt19937{12345u};
    auto next = [&] {
      return (float(random() % 20001u) - 10000.0f) / 10000.0f;
    };
    for (Index row = 0; row < shape.m; ++row)
      for (Index inner = 0; inner < shape.k; ++inner)
        aHost[Matrix{row, inner}] = next();
    for (Index inner = 0; inner < shape.k; ++inner)
      for (Index column = 0; column < shape.n; ++column)
        bHost[Matrix{inner, column}] = next();
    auto const points = reference(aHost, bHost, shape, options.selfTest);
    auto a = alpaka::onHost::allocLike(device, aHost);
    auto b = alpaka::onHost::allocLike(device, bHost);
    auto c = alpaka::onHost::allocLike(device, cHost);
    alpaka::onHost::memcpy(queue, a, aHost);
    alpaka::onHost::memcpy(queue, b, bHost);
    alpaka::onHost::wait(queue);
    auto const vectorLoads =
        a.getPitches()[0u] % 16u == 0u && b.getPitches()[0u] % 16u == 0u &&
        reinterpret_cast<std::uintptr_t>(a.data()) % 16u == 0u &&
        reinterpret_cast<std::uintptr_t>(b.data()) % 16u == 0u;

    auto validate = [&](std::string const &label) {
      alpaka::onHost::memcpy(queue, cHost, c);
      alpaka::onHost::wait(queue);
      check(cHost, points, label);
    };
    auto blockCounts = std::vector<Index>{};
    Index legal{};
    for (Index variant = 0; variant < tiles.size(); ++variant) {
      if (threads > properties.maxThreadsPerBlock ||
          (properties.sharedMemPerBlockBytes &&
           tiles[variant].sharedBytes() > properties.sharedMemPerBlockBytes))
        continue;
      ++legal;
      auto const blocks = tiles[variant].blocks(shape);
      if (std::find(blockCounts.begin(), blockCounts.end(), blocks) ==
          blockCounts.end())
        blockCounts.push_back(blocks);
      alpaka::onHost::memset(queue, c, 0xff);
      launch(variant, device, queue, executor, a, b, c, shape, vectorLoads,
             threads);
      validate("variant " + std::to_string(variant));
    }
    if (!legal)
      throw std::runtime_error{"Device cannot run any GEMM variant"};

    auto vendorLaunch = [&] {
#if ALPAKA_TUNE_GEMM_VENDOR
      // Instantiate BLAS dispatch only for enabled backend/library pairs.
      if constexpr (!vendorName<ALPAKA_TYPEOF(device)>().empty()) {
        alpaka::blas::onHost::gemm(
            queue, 1.0f, a, b, 0.0f, c,
            alpaka::blas::Options{.precision = alpaka::blas::Precision::exact,
                                  .algorithm =
                                      alpaka::blas::Algorithm::backendDefault});
      }
#endif
    };
    if (useVendor) {
      vendorLaunch();
      validate(std::string{vendor});
    }

    auto const variantTunable = variantName(
        alpakaTune::CVals<0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u>{});
    auto const blocksTunable =
        alpakaTune::numBlocks(alpakaTune::RVals<Index>{blockCounts});
    auto tunables = alpakaTune::constrain(
        alpakaTune::TunableBundle{variantTunable, blocksTunable},
        alpakaTune::restrict(
            variantName, alpakaTune::numBlocks,
            [shape, properties, threads](Index variant, Index blocks) {
              auto const tile = tiles.at(variant);
              return blocks == tile.blocks(shape) &&
                     threads <= properties.maxThreadsPerBlock &&
                     (!properties.sharedMemPerBlockBytes ||
                      tile.sharedBytes() <= properties.sharedMemPerBlockBytes);
            }));
    auto config = alpakaTune::TunerConfig{};
    config.strategy = alpakaTune::StrategyKind::exhaustive;
    config.selection = alpakaTune::SelectionPolicy::fixed;
    config.runsPerCandidate = options.samplesPerCandidate;
    config.minimumRunsPerCandidate = options.samplesPerCandidate;
    config.historyWindowSize = options.samplesPerCandidate;
    config.mannWhitneyEarlyStop = false;
    config.maximumConsecutiveStrategyRetries =
        tiles.size() * blockCounts.size();
    config.maximumExecutions = legal * (options.samplesPerCandidate + 1u) + 1u;
    config.queue = alpakaTune::QueueConfig{
        .warmupRuns = 1u,
        .noiseCancellationWindow = legal,
        .maxConsecutiveRuns = options.samplesPerCandidate + 1u};
    config.history.file = options.history;
    config.history.read = options.reuseHistory;
    config.history.write = options.history.has_value();
    config.completeHistory.read = false;
    config.completeHistory.write = false;
    if (options.reuseHistory) {
      config.exploration = alpakaTune::ExplorationPolicy::offline;
      config.maximumExecutions.reset();
    }
    auto const identity = "gemm/v1/fp32/" + std::to_string(shape.m) + "x" +
                          std::to_string(shape.n) + "x" +
                          std::to_string(shape.k);
    auto tuner =
        alpakaTune::makeTuner(config, tunables, device, executor, identity);
    auto const prototype =
        alpaka::KernelBundle{MatmulKernel{},
                             a,
                             b,
                             c,
                             shape,
                             vectorLoads,
                             alpakaTune::markTunable(variantName)};
    auto const launchSpec = alpaka::onHost::ThreadSpec{
        tiles.front().blocks(shape), threads, executor};
    auto const tuneStart = Clock::now();
    std::size_t launches{};
    do {
      if (++launches > legal * (options.samplesPerCandidate + 1u) + 2u)
        throw std::runtime_error{
            "GEMM tuning exceeded its bounded launch count"};
      tuner.enqueue(queue, launchSpec, prototype);
    } while (!tuner.completed());
    alpaka::onHost::wait(queue);
    auto const tuningSeconds =
        std::chrono::duration<double>(Clock::now() - tuneStart).count();
    if (!options.reuseHistory && tuner.info().measuredCandidateCount != legal)
      throw std::runtime_error{
          "Tuning stopped before measuring every legal variant"};
    // CVals are ordered 0..11; the first normalized coordinate identifies that
    // ordinal. Verify it rather than depending on Cartesian candidate ordering.
    auto const normalized = tuner.bestConfiguration().at(0u);
    auto const winner =
        static_cast<Index>(std::lround(normalized * (tiles.size() - 1u)));
    if (winner >= tiles.size())
      throw std::runtime_error{"Invalid normalized GEMM winner"};
    validate("tuned kernel");

    auto prefix = [&]() {
      return quote(device.getName()) + ',' + quote(spec.getApi().getName()) +
             ',' + quote(alpaka::onHost::demangledName(executor)) + ',' +
             std::to_string(shape.m) + ',' + std::to_string(shape.n) + ',' +
             std::to_string(shape.k) + ',';
    };
    if (output.tuning.is_open())
      for (auto const &entry : tuner.history()) {
        output.tuning << prefix() << entry.candidateIndex << ','
                      << entry.executionIndex << ',' << entry.measured << ',';
        if (entry.runtimeSeconds)
          output.tuning << *entry.runtimeSeconds;
        output.tuning << '\n';
      }

    std::array<std::vector<double>, 3> durations, wallDurations;
    auto const count = useVendor ? 3u : 2u;
    auto implementationLaunch = [&](Index implementation) {
      if (implementation == 2u)
        vendorLaunch();
      else
        launch(implementation == 0u ? 0u : winner, device, queue, executor, a,
               b, c, shape, vectorLoads, threads);
    };
    for (Index implementation = 0; implementation < count; ++implementation)
      for (Index warmup = 0; warmup < 3u; ++warmup)
        implementationLaunch(implementation);
    alpaka::onHost::wait(queue);
    auto start = device.makeEvent(alpaka::timing::enabled);
    auto end = device.makeEvent(alpaka::timing::enabled);
    constexpr std::array labels{"default", "tuned", "vendor"};
    for (Index repetition = 0; repetition < options.repetitions; ++repetition) {
      // Rotate order to reduce systematic clock/temperature drift.
      for (Index offset = 0; offset < count; ++offset) {
        auto const implementation = (repetition + offset) % count;
        auto const wallStart = Clock::now();
        queue.enqueue(start);
        implementationLaunch(implementation);
        queue.enqueue(end);
        alpaka::onHost::wait(queue);
        auto const seconds = alpaka::onHost::getElapsedTime(start, end).count();
        auto const wallSeconds =
            std::chrono::duration<double>(Clock::now() - wallStart).count();
        if (!(seconds > 0.0) || !std::isfinite(seconds))
          throw std::runtime_error{"Invalid GEMM timing"};
        durations[implementation].push_back(seconds);
        wallDurations[implementation].push_back(wallSeconds);
        if (output.samples.is_open())
          output.samples << prefix() << labels[implementation] << ','
                         << repetition << ',' << seconds << ',' << wallSeconds
                         << '\n';
      }
    }
    for (Index implementation = 0; implementation < count; ++implementation) {
      alpaka::onHost::memset(queue, c, 0xff);
      implementationLaunch(implementation);
      validate(labels[implementation]);
    }
    auto const baseline = median(durations[0]);
    auto const optimized = median(durations[1]);
    auto const baselineWall = median(wallDurations[0]);
    auto const optimizedWall = median(wallDurations[1]);
    auto const payback = breakEven(tuningSeconds, baseline, optimized);
    auto const wallPayback =
        breakEven(tuningSeconds, baselineWall, optimizedWall);
    auto const improvement = 100.0 * (1.0 - optimized / baseline);
    auto const defaultInterval = speedupInterval(durations[0], durations[1]);
    auto const vendorInterval =
        useVendor ? speedupInterval(durations[2], durations[1]) : std::nullopt;
    auto const flops = 2.0 * shape.m * shape.n * shape.k;
    std::cout << std::setprecision(5) << "  " << shape.m << 'x' << shape.n
              << 'x' << shape.k << ": default " << baseline * 1e3
              << " ms; tuned " << optimized * 1e3 << " ms ("
              << flops / optimized / 1e9 << " GFLOP/s), " << improvement
              << "% time reduction, " << baseline / optimized
              << "x default; variant " << winner << "; " << legal
              << " legal variants\n";
    if (defaultInterval)
      std::cout << "    default speedup 95% bootstrap interval: ["
                << (*defaultInterval)[0] << ", " << (*defaultInterval)[1]
                << "]\n";
    if (useVendor)
      std::cout << "    " << vendor << ": " << median(durations[2]) * 1e3
                << " ms; tuned speedup " << median(durations[2]) / optimized
                << "x vendor\n";
    std::cout << "    tuning " << tuningSeconds << " s / " << launches
              << " launches; break-even ";
    if (payback)
      std::cout << *payback << " reuses (queue time)";
    else
      std::cout << "not reached (queue time)";
    std::cout << "; wall-time break-even ";
    if (wallPayback)
      std::cout << *wallPayback;
    else
      std::cout << "not reached";
    std::cout << '\n';

    if (output.summary.is_open()) {
      auto const tile = tiles[winner];
      output.summary << prefix() << "fp32_exact,exhaustive," << legal << ','
                     << winner << ',' << tile.m << ',' << tile.n << ','
                     << tile.k << ',' << tile.rowsPerThread << ','
                     << tile.columnsPerThread << ',' << tile.stages << ','
                     << tuner.loadedFromCache() << ',' << launches << ','
                     << tuningSeconds << ',' << baseline << ',' << optimized
                     << ',';
      if (useVendor)
        output.summary << median(durations[2]);
      output.summary << ',' << quote(useVendor ? vendor : std::string_view{})
                     << ',' << improvement << ',' << baseline / optimized
                     << ',';
      if (useVendor)
        output.summary << median(durations[2]) / optimized;
      output.summary << ',';
      if (payback)
        output.summary << *payback;
      output.summary << ',' << baselineWall << ',' << optimizedWall << ',';
      if (useVendor)
        output.summary << median(wallDurations[2]);
      output.summary << ',';
      if (wallPayback)
        output.summary << *wallPayback;
      for (auto interval : {defaultInterval, vendorInterval}) {
        output.summary << ',';
        if (interval)
          output.summary << (*interval)[0];
        output.summary << ',';
        if (interval)
          output.summary << (*interval)[1];
      }
      output.summary << '\n';
    }
    if (options.selfTest)
      std::cout << "    correctness: all output elements checked for every "
                   "legal variant\n";
  }
}
} // namespace alpakaTune::benchmarks::matmul

int main(int argc, char **argv) {
  using namespace alpakaTune::benchmarks::matmul;
  try {
    if (!alpakaTune::consumeBackendOptions(argc, argv))
      return EXIT_FAILURE;
    auto const options = parseOptions(argc, argv);
    auto output = Output{options};
    auto ran = false;
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
      throw std::runtime_error{
          "No device matches the selected backend/executor"};
    return EXIT_SUCCESS;
  } catch (std::exception const &exception) {
    std::cerr << "GEMM: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
}
