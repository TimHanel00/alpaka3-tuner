// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#include "Matmul.hpp"
#include "NativeCublas.hpp"

#include <alpakaTune/BackendSelection.hpp>

#include <array>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace alpakaTune::example::matmul {
struct Options {
  std::vector<Shape> shapes;
  Index repetitions{31u};
  Index device{};
  bool selfTest{};
  bool cublas{ALPAKA_TUNE_MATMUL_CUBLAS != 0};
  std::optional<std::string> csv;
};

auto positive(std::string_view text) -> Index {
  Index value{};
  auto const [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size() || value == 0u)
    throw std::invalid_argument{"Expected a positive integer"};
  return value;
}

auto parseOptions(int argc, char **argv) -> Options {
  auto options = Options{};
  for (int i = 1; i < argc; ++i) {
    auto const argument = std::string_view{argv[i]};
    auto value = [&]() -> std::string_view {
      if (++i == argc)
        throw std::invalid_argument{"Missing option value"};
      return argv[i];
    };
    if (argument == "--shape") {
      auto const text = value();
      auto const first = text.find('x');
      auto const second = text.find('x', first == text.npos ? 0u : first + 1u);
      if (first == text.npos || second == text.npos)
        throw std::invalid_argument{"--shape expects MxNxK"};
      auto const shape =
          Shape{positive(text.substr(0u, first)),
                positive(text.substr(first + 1u, second - first - 1u)),
                positive(text.substr(second + 1u))};
      if (std::max({shape.m, shape.n, shape.k}) > (1u << 20u))
        throw std::invalid_argument{"Dimensions must not exceed 1048576"};
      options.shapes.push_back(shape);
    } else if (argument == "--repetitions") {
      options.repetitions = positive(value());
      if (options.repetitions > 10000u)
        throw std::invalid_argument{"At most 10000 repetitions"};
    } else if (argument == "--device") {
      auto const text = value();
      options.device = text == "0" ? 0u : positive(text);
    } else if (argument == "--csv")
      options.csv = value();
    else if (argument == "--self-test")
      options.selfTest = true;
    else if (argument == "--no-cublas")
      options.cublas = false;
    else if (argument == "--help") {
      std::cout << "Compiled matmul winners; no tuning at runtime\n"
                   "  --backend api:deviceKind --executor name --device INDEX\n"
                   "  --shape MxNxK (repeatable) --repetitions N --csv FILE\n"
                   "  --no-cublas --self-test\n";
      std::exit(EXIT_SUCCESS);
    } else
      throw std::invalid_argument{"Unknown option: " + std::string{argument}};
  }
  return options;
}

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
    for (Index k = 0u; k < shape.k; ++k) {
      auto const term =
          double{a[alpaka::Vec{row, k}]} * double{b[alpaka::Vec{k, column}]};
      expected += term;
      magnitude += std::abs(term);
    }
    auto const epsilon = double{std::numeric_limits<float>::epsilon()};
    auto const gamma = shape.k * epsilon / (1.0 - shape.k * epsilon);
    points.push_back({row, column, expected, 3.0 * gamma * magnitude + 2.0e-6});
  };
  if (full) {
    for (Index row = 0u; row < shape.m; ++row)
      for (Index column = 0u; column < shape.n; ++column)
        add(row, column);
  } else {
    for (auto row : {0u, shape.m / 2u, shape.m - 1u})
      for (auto column : {0u, shape.n / 2u, shape.n - 1u})
        add(row, column);
    auto random = std::mt19937{42u};
    for (Index sample = 0u; sample < 64u; ++sample)
      add(random() % shape.m, random() % shape.n);
  }
  return points;
}

void check(alpaka::concepts::IBuffer auto const &c,
           std::vector<ReferencePoint> const &points) {
  for (auto point : points) {
    auto const actual = double{c[alpaka::Vec{point.row, point.column}]};
    if (!std::isfinite(actual) ||
        std::abs(actual - point.value) > point.tolerance)
      throw std::runtime_error{"Incorrect result at C[" +
                               std::to_string(point.row) + "," +
                               std::to_string(point.column) + "]"};
  }
}

auto median(std::vector<double> values) -> double {
  std::sort(values.begin(), values.end());
  auto const middle = values.size() / 2u;
  return values.size() % 2u ? values[middle]
                            : (values[middle - 1u] + values[middle]) / 2.0;
}

void run(alpaka::concepts::BackendSpec auto spec, Options const &options,
         std::ostream *csv) {
  auto const executor = alpaka::getExecutor(spec);
  auto device =
      alpaka::onHost::makeDeviceSelector(alpaka::onHost::DeviceSpec{spec})
          .makeDevice(options.device);
  auto queue =
      device.makeQueue(alpaka::queueKind::nonBlocking, alpaka::timing::enabled);
  using Api = ALPAKA_TYPEOF(device.getApi());
  constexpr bool cuda = std::same_as<Api, alpaka::api::Cuda>;
  auto const useCublas = options.cublas && cuda;
  auto native = std::optional<NativeCublas<Api>>{};
  if (useCublas)
    native.emplace(queue);
  std::cout << device.getName() << " / " << device.getApi().getName() << " / "
            << alpaka::onHost::demangledName(executor) << '\n';
  Index ompThreads{1u};
#if ALPAKA_OMP
  if constexpr (std::same_as<ALPAKA_TYPEOF(executor),
                             alpaka::exec::CpuOmpBlocks>) {
#pragma omp parallel
    {
#pragma omp single
      ompThreads = static_cast<Index>(omp_get_num_threads());
    }
    if (ompThreads != static_cast<Index>(omp_get_max_threads()))
      throw std::runtime_error{
          "Disable dynamic OpenMP teams for winner lookup"};
    std::cout << "OpenMP threads: " << ompThreads << " / device CPU units: "
              << device.getDeviceProperties().multiProcessorCount << '\n';
  }
#endif

  auto evaluate = [&](Shape shape, bool full, auto const &launches) {
    using Matrix = alpaka::Vec<std::size_t, 2u>;
    auto ah = alpaka::onHost::allocHost<float>(Matrix{shape.m, shape.k});
    auto bh = alpaka::onHost::allocHost<float>(Matrix{shape.k, shape.n});
    auto ch = alpaka::onHost::allocHost<float>(Matrix{shape.m, shape.n});
    auto random = std::mt19937{12345u};
    auto next = [&] {
      return (float(random() % 20001u) - 10000.0f) / 10000.0f;
    };
    for (Index row = 0u; row < shape.m; ++row)
      for (Index k = 0u; k < shape.k; ++k)
        ah[Matrix{row, k}] = next();
    for (Index k = 0u; k < shape.k; ++k)
      for (Index column = 0u; column < shape.n; ++column)
        bh[Matrix{k, column}] = next();
    auto const points = reference(ah, bh, shape, full);
    auto a = alpaka::onHost::allocLike(device, ah);
    auto b = alpaka::onHost::allocLike(device, bh);
    auto c = alpaka::onHost::allocLike(device, ch);
    alpaka::onHost::memcpy(queue, a, ah);
    alpaka::onHost::memcpy(queue, b, bh);
    alpaka::onHost::wait(queue);
    launches(a, b, c, [&](auto const &launch) {
      alpaka::onHost::memset(queue, c, 0xff);
      launch();
      alpaka::onHost::memcpy(queue, ch, c);
      alpaka::onHost::wait(queue);
      check(ch, points);
    });
  };

  if (options.selfTest) {
    auto verifyWinner = [](auto selected) {
      using Expected = decltype(selected);
      constexpr auto shape = Expected::shape;
      std::uint32_t matches{};
      if (!selectConfiguration(
              "NVIDIA A30", "Cuda", "alpaka::exec::GpuCuda", shape,
              [&](auto found) {
                if constexpr (!std::same_as<decltype(found), Expected>)
                  throw std::runtime_error{
                      "Wrong winner for " + std::to_string(shape.m) + "x" +
                      std::to_string(shape.n) + "x" + std::to_string(shape.k)};
                ++matches;
              }) ||
          matches != 1u)
        throw std::runtime_error{"Winner lookup failed"};
    };
    std::apply([&](auto... selected) { (verifyWinner(selected), ...); },
               winners);
    auto verifyCpuWinner = [](auto selected) {
      using Expected = decltype(selected);
      Index matches{};
      if (!selectConfiguration(
              "AMD EPYC 7713 64-Core Processor  ", "Host",
              "alpaka::exec::CpuOmpBlocks", Expected::shape,
              [&](auto actual) {
                if constexpr (!std::same_as<decltype(actual), Expected>)
                  throw std::runtime_error{"Wrong CPU winner type"};
                ++matches;
              },
              {Expected::numProcessingUnits, Expected::numOmpThreads}) ||
          matches != 1u)
        throw std::runtime_error{"CPU winner lookup failed"};
    };
    std::apply([&](auto... selected) { (verifyCpuWinner(selected), ...); },
               cpuWinners);
    auto reject = [](auto) { throw std::runtime_error{"Unexpected winner"}; };
    for (auto context : {CpuContext{64, 128}, CpuContext{128, 4}})
      if (selectConfiguration("AMD EPYC 7713 64-Core Processor", "Host",
                              "alpaka::exec::CpuOmpBlocks", {512, 512, 512},
                              reject, context))
        throw std::runtime_error{"Unmatched CPU context selected a winner"};
    if (selectConfiguration("other CPU", "Host", "alpaka::exec::CpuOmpBlocks",
                            {512, 512, 512}, reject, {128, 128}) ||
        selectConfiguration("AMD EPYC 7713 64-Core Processor", "Host",
                            "alpaka::exec::CpuSerial", {512, 512, 512}, reject,
                            {128, 1}))
      throw std::runtime_error{"Unmatched CPU device selected a winner"};
    for (auto key : {std::array<std::string_view, 3>{"other GPU", "Cuda",
                                                     "alpaka::exec::GpuCuda"},
                     std::array<std::string_view, 3>{"NVIDIA A30", "Hip",
                                                     "alpaka::exec::GpuCuda"},
                     std::array<std::string_view, 3>{"NVIDIA A30", "Cuda",
                                                     "other executor"}})
      if (selectConfiguration(key[0], key[1], key[2], {512, 512, 512}, reject))
        throw std::runtime_error{"Unmatched device selected a winner"};
    if (selectConfiguration("NVIDIA A30", "Cuda", "alpaka::exec::GpuCuda",
                            {17, 29, 19}, reject))
      throw std::runtime_error{"Unmatched shape selected a winner"};

    auto cases = []<std::size_t... Layouts>(std::index_sequence<Layouts...>) {
      return std::tuple{std::integral_constant<Index, Layouts>{}...};
    }(std::make_index_sequence<tiles.size()>{});
    auto testShape = [&]<Index M, Index N, Index K>(
                         Configuration<M, N, K, 0, 256, 1>) {
      evaluate(
          {M, N, K}, true,
          [&](auto const &a, auto const &b, auto &c, auto validate) {
            validate([&] { enqueueBaseline(queue, executor, a, b, c); });
            validate([&] {
              if (enqueueMatmul(queue, executor, a, b, c))
                throw std::runtime_error{"Small case should use fallback"};
            });
            std::apply(
                [&](auto... layout) {
                  (
                      [&] {
                        constexpr auto id = decltype(layout)::value;
                        constexpr auto threads = tiles[id].logicalThreads();
                        using Full = Configuration<M, N, K, id, threads,
                                                   tiles[id].blocks({M, N, K})>;
                        using Bounded = Configuration<M, N, K, id, threads, 1>;
                        validate([&] {
                          enqueueCompiled(Full{}, queue, executor, a, b, c);
                        });
                        validate([&] {
                          enqueueCompiled(Bounded{}, queue, executor, a, b, c);
                        });
                      }(),
                      ...);
                },
                cases);
            if (useCublas)
              validate([&] { native->gemm(queue, a, b, c, {M, N, K}); });
          });
    };
    testShape(Configuration<17, 29, 19, 0, 256, 1>{});
    testShape(Configuration<64, 128, 32, 0, 256, 1>{});
    testShape(Configuration<129, 131, 35, 0, 256, 1>{});
    std::cout << "Winner lookup, baseline, fallback, all selected layouts, "
                 "full and bounded grids: passed\n";
    return;
  }

  auto shapes = options.shapes;
  if (shapes.empty()) {
    if constexpr (cuda)
      std::apply(
          [&](auto... selected) {
            (shapes.push_back(decltype(selected)::shape), ...);
          },
          winners);
    else {
      CpuContext context{
          static_cast<Index>(device.getDeviceProperties().multiProcessorCount),
          ompThreads};
      std::apply(
          [&](auto... selected) {
            (
                [&] {
                  auto const shape = decltype(selected)::shape;
                  if (selectConfiguration(
                          device.getName(), device.getApi().getName(),
                          alpaka::onHost::demangledName(executor), shape,
                          [](auto) {}, context))
                    shapes.push_back(shape);
                }(),
                ...);
          },
          winners);
      if (shapes.empty())
        shapes.push_back({64, 128, 32});
    }
  }
  for (auto shape : shapes) {
    evaluate(
        shape, false,
        [&](auto const &a, auto const &b, auto &c, auto validate) {
          bool selected{};
          auto launch = [&](Index implementation) {
            if (implementation == 0u)
              enqueueBaseline(queue, executor, a, b, c);
            else if (implementation == 1u)
              selected = enqueueMatmul(queue, executor, a, b, c);
            else
              native->gemm(queue, a, b, c, shape);
          };
          auto const count = useCublas ? 3u : 2u;
          for (Index implementation = 0u; implementation < count;
               ++implementation) {
            validate([&] { launch(implementation); });
            for (Index warmup = 0u; warmup < 3u; ++warmup)
              launch(implementation);
          }
          alpaka::onHost::wait(queue);
          auto start = device.makeEvent(alpaka::timing::enabled);
          auto end = device.makeEvent(alpaka::timing::enabled);
          std::array<std::vector<double>, 3> times;
          constexpr std::array labels{"default matmul", "tuned matmul",
                                      "strict FP32 cuBLAS"};
          for (Index repetition = 0u; repetition < options.repetitions;
               ++repetition)
            for (Index offset = 0u; offset < count; ++offset) {
              auto const implementation = (repetition + offset) % count;
              queue.enqueue(start);
              launch(implementation);
              queue.enqueue(end);
              alpaka::onHost::wait(queue);
              auto const ms =
                  alpaka::onHost::getElapsedTime(start, end).count() * 1000.0;
              if (!(ms > 0.0) || !std::isfinite(ms))
                throw std::runtime_error{"Invalid runtime measurement"};
              times[implementation].push_back(ms);
              if (csv)
                *csv << std::quoted(device.getName()) << ',' << shape.m << ','
                     << shape.n << ',' << shape.k << ','
                     << labels[implementation] << ',' << repetition << ',' << ms
                     << ',' << (implementation == 1u && selected) << ','
                     << (cuda
                             ? 0u
                             : device.getDeviceProperties().multiProcessorCount)
                     << ',' << ompThreads << '\n';
            }
          std::cout << shape.m << 'x' << shape.n << 'x' << shape.k << " / "
                    << (selected ? "compiled winner" : "baseline fallback");
          for (Index i = 0u; i < count; ++i)
            std::cout << " / " << labels[i] << ": " << median(times[i])
                      << " ms";
          std::cout << '\n';
        });
  }
}
} // namespace alpakaTune::example::matmul

int main(int argc, char **argv) {
  using namespace alpakaTune::example::matmul;
  try {
    if (!alpakaTune::consumeBackendOptions(argc, argv))
      return EXIT_FAILURE;
    auto const options = parseOptions(argc, argv);
    auto csv = std::ofstream{};
    if (options.csv) {
      csv.open(*options.csv);
      if (!csv)
        throw std::runtime_error{"Cannot open CSV"};
      csv << "device,m,n,k,implementation,repetition,runtime_ms,winner_"
             "selected,cpu_units,omp_threads\n"
          << std::setprecision(12);
    }
    bool ran{};
    alpaka::onHost::executeForEach(
        [&](alpaka::concepts::BackendSpec auto spec) {
          if (alpakaTune::backendSelected(spec) &&
              alpaka::onHost::makeDeviceSelector(
                  alpaka::onHost::DeviceSpec{spec})
                  .isAvailable()) {
            run(spec, options, options.csv ? &csv : nullptr);
            ran = true;
          }
          return EXIT_SUCCESS;
        },
        alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs,
                                    alpaka::exec::enabledExecutors));
    if (!ran)
      throw std::runtime_error{
          "No available device matches the selected backend/executor"};
    if (options.csv) {
      csv.close();
      if (!csv)
        throw std::runtime_error{"Failed to write CSV"};
    }
    return EXIT_SUCCESS;
  } catch (std::exception const &error) {
    std::cerr << "matmul: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
