/* Copyright 2026 Tim Hanel
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>

#include <tuning.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace alpaka::example::matrixMultiplication {
using Index = std::size_t;
using Scalar = float;

inline constexpr auto blockRowsTunable = ALPAKA_TUNE_TUNABLE("blockRows");
inline constexpr auto blockColumnsTunable = ALPAKA_TUNE_TUNABLE("blockColumns");
inline constexpr auto kTileTunable = ALPAKA_TUNE_TUNABLE("kTile");
inline constexpr auto rowsPerThreadTunable =
    ALPAKA_TUNE_TUNABLE("rowsPerThread");
inline constexpr auto simdWidthTunable = ALPAKA_TUNE_TUNABLE("simdWidth");
inline constexpr auto tilesPerGroupTunable =
    ALPAKA_TUNE_TUNABLE("tilesPerGroup");

struct TiledMatrixMultiplicationKernel {
  template <typename T_Acc, typename T_View>
  ALPAKA_FN_ACC auto
  operator()(T_Acc const &acc, T_View matrixA, T_View matrixB, T_View matrixC,
             Index const rows, Index const columns, Index const innerDimension,
             Index const tilesPerGroup, auto const blockRowsConstant,
             auto const blockColumnsConstant, auto const kTileConstant,
             auto const rowsPerThreadConstant,
             auto const simdWidthConstant) const -> void {
    constexpr Index blockRows = decltype(blockRowsConstant)::value;
    constexpr Index blockColumns = decltype(blockColumnsConstant)::value;
    constexpr Index kTile = decltype(kTileConstant)::value;
    constexpr Index rowsPerThread = decltype(rowsPerThreadConstant)::value;
    constexpr std::uint32_t simdWidth = decltype(simdWidthConstant)::value;
    static_assert(blockRows % rowsPerThread == 0u);
    static_assert(blockColumns % simdWidth == 0u);

    constexpr auto sharedAExtent = CVec<Index, blockRows, kTile>{};
    // Padding the contiguous direction by one element avoids the most common
    // shared-memory bank-conflict pattern for transposed/strided accesses.
    constexpr auto sharedBExtent = CVec<Index, kTile, blockColumns + 1u>{};
    constexpr auto microTileExtent =
        CVec<Index, blockRows / rowsPerThread, blockColumns / simdWidth>{};

    auto sharedA =
        onAcc::declareSharedMdArray<Scalar, uniqueId()>(acc, sharedAExtent);
    auto sharedB =
        onAcc::declareSharedMdArray<Scalar, uniqueId()>(acc, sharedBExtent);

    auto const tileRows = divCeil(rows, blockRows);
    auto const tileColumns = divCeil(columns, blockColumns);
    auto const tileCount = tileRows * tileColumns;
    auto const groupCount = divCeil(tileCount, tilesPerGroup);

    for (auto [linearGroup] : onAcc::makeIdxMap(
             acc, onAcc::worker::blocksInGrid, IdxRange{groupCount})) {
      // Runtime grouping changes how many adjacent output tiles a
      // scheduled block processes before it advances to its next
      // group. It adds scheduling coverage without instantiating a
      // new kernel type.
      for (Index tileInGroup = 0u; tileInGroup < tilesPerGroup; ++tileInGroup) {
        auto const linearTile = linearGroup * tilesPerGroup + tileInGroup;
        if (linearTile >= tileCount)
          break;
        auto const tileRow = linearTile / tileColumns;
        auto const tileColumn = linearTile % tileColumns;
        auto const rowBase = tileRow * blockRows;
        auto const columnBase = tileColumn * blockColumns;

        auto const threadExtents =
            acc.getExtentsOf(onAcc::origin::block, onAcc::unit::threads);
        auto const threadIndex =
            linearize(threadExtents, acc.getIdxWithin(onAcc::origin::block,
                                                      onAcc::unit::threads));
        auto const threadCount = threadExtents.product();
        using Pack = Simd<Scalar, simdWidth>;
        constexpr auto microTileCount = microTileExtent.product();
        for (Index roundBegin = 0u; roundBegin < microTileCount;
             roundBegin += threadCount) {
          auto const linearMicroTile = roundBegin + threadIndex;
          auto const active = linearMicroTile < microTileCount;
          auto const microTileRow = linearMicroTile / microTileExtent[1u];
          auto const microTileColumn = linearMicroTile % microTileExtent[1u];
          Pack accumulators[rowsPerThread];
          for (Index localRow = 0u; localRow < rowsPerThread; ++localRow)
            accumulators[localRow] = Pack{[](auto) { return Scalar{0.0f}; }};

          for (Index kBase = 0u; kBase < innerDimension; kBase += kTile) {
            for (auto [linear] :
                 onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock,
                                   IdxRange{blockRows * kTile})) {
              auto const localRow = linear / kTile;
              auto const localK = linear % kTile;
              auto const globalRow = rowBase + localRow;
              auto const globalK = kBase + localK;
              sharedA[Vec{localRow, localK}] =
                  globalRow < rows && globalK < innerDimension
                      ? matrixA[globalRow * innerDimension + globalK]
                      : Scalar{0.0f};
            }
            for (auto [linear] :
                 onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock,
                                   IdxRange{kTile * blockColumns})) {
              auto const localK = linear / blockColumns;
              auto const localColumn = linear % blockColumns;
              auto const globalK = kBase + localK;
              auto const globalColumn = columnBase + localColumn;
              sharedB[Vec{localK, localColumn}] =
                  globalK < innerDimension && globalColumn < columns
                      ? matrixB[globalK * columns + globalColumn]
                      : Scalar{0.0f};
            }

            onAcc::syncBlockThreads(acc);

            if (active) {
              auto const firstLocalRow = microTileRow * rowsPerThread;
              auto const firstLocalColumn = microTileColumn * simdWidth;
              for (Index localK = 0u; localK < kTile; ++localK) {
                auto const packedB = Pack{[&](auto lane) {
                  return sharedB[Vec{localK,
                                     firstLocalColumn + decltype(lane)::value}];
                }};
                for (Index localRow = 0u; localRow < rowsPerThread;
                     ++localRow) {
                  auto const valueA =
                      sharedA[Vec{firstLocalRow + localRow, localK}];
                  auto const broadcastA =
                      Pack{[valueA](auto) { return valueA; }};
                  accumulators[localRow] += broadcastA * packedB;
                }
              }
            }

            onAcc::syncBlockThreads(acc);
          }

          if (active) {
            for (Index localRow = 0u; localRow < rowsPerThread; ++localRow) {
              auto const globalRow =
                  rowBase + microTileRow * rowsPerThread + localRow;
              auto const globalColumnBase =
                  columnBase + microTileColumn * simdWidth;
              if (globalRow < rows) {
                for (std::uint32_t lane = 0u; lane < simdWidth; ++lane) {
                  auto const globalColumn = globalColumnBase + lane;
                  if (globalColumn < columns)
                    matrixC[globalRow * columns + globalColumn] =
                        accumulators[localRow][lane];
                }
              }
            }
          }
        }
      }
    }
  }
};

auto launchCounts() -> std::vector<Vec<Index, 1u>> {
  return {Vec{Index{1u}}, Vec{Index{8u}}, Vec{Index{32u}}, Vec{Index{128u}}};
}

auto example(auto const deviceSpec, auto const executor, Index rows,
             Index columns, Index innerDimension, Index minimumRuns) -> int {
  using Vector = Vec<Index, 1u>;

  auto selector = onHost::makeDeviceSelector(deviceSpec);
  onHost::Device device = selector.makeDevice(0);
  onHost::Queue queue = device.makeQueue();

  auto hostA = onHost::allocHost<Scalar>(Vector{rows * innerDimension});
  auto hostB = onHost::allocHost<Scalar>(Vector{innerDimension * columns});
  auto hostC = onHost::allocHost<Scalar>(Vector{rows * columns});
  for (Index row = 0u; row < rows; ++row)
    for (Index column = 0u; column < innerDimension; ++column)
      hostA[row * innerDimension + column] =
          static_cast<Scalar>(static_cast<int>((row * 3u + column * 5u) % 17u) -
                              8) /
          Scalar{8.0f};
  for (Index row = 0u; row < innerDimension; ++row)
    for (Index column = 0u; column < columns; ++column)
      hostB[row * columns + column] =
          static_cast<Scalar>(static_cast<int>((row * 7u + column * 2u) % 19u) -
                              9) /
          Scalar{9.0f};

  auto deviceA = onHost::allocLike(device, hostA);
  auto deviceB = onHost::allocLike(device, hostB);
  auto deviceC = onHost::allocLike(device, hostC);
  onHost::memcpy(queue, deviceA, hostA);
  onHost::memcpy(queue, deviceB, hostB);
  onHost::wait(queue);

  constexpr Index largestBlockRows = 64u;
  constexpr Index largestBlockColumns = 64u;
  auto const tileCount =
      divCeil(rows, largestBlockRows) * divCeil(columns, largestBlockColumns);
  auto const frameSpec =
      onHost::FrameSpec{Vector{tileCount}, Vector{256u}, executor};
  auto tunables = alpakaTune::constrain(
      alpakaTune::TunableBundle{
          alpakaTune::numFrames(alpakaTune::RVals<Vector>{launchCounts()}),
          alpakaTune::frameExtent(alpakaTune::RVals<Vector>{
              Vector{64u}, Vector{128u}, Vector{256u}}),
          tilesPerGroupTunable(alpakaTune::RVals<Index>{1u, 2u, 4u, 8u, 16u}),
          blockRowsTunable(alpakaTune::CVals<16u, 32u, 64u>{}),
          blockColumnsTunable(alpakaTune::CVals<16u, 32u, 64u>{}),
          kTileTunable(alpakaTune::CVals<8u, 16u>{}),
          rowsPerThreadTunable(alpakaTune::CVals<1u, 2u>{}),
          simdWidthTunable(alpakaTune::CVals<1u, 2u>{})},
      alpakaTune::restrict(blockRowsTunable, rowsPerThreadTunable,
                           [](auto block, auto rowsPerThread) {
                             return block % rowsPerThread == 0u;
                           }),
      alpakaTune::restrict(
          blockColumnsTunable, simdWidthTunable,
          [](auto block, auto width) { return block % width == 0u; }));
  auto tuner = alpakaTune::makeTuner(tunables, device, executor,
                                     "matrixMultiplication/tiledSimd");
  // 4 numFrames * 3 frameExtent * 5 tilesPerGroup runtime choices,
  // multiplied by 3 * 3 * 2 * 2 * 2 = 72 compile-time variants.
  constexpr Index expectedCandidateCount = 4u * 3u * 5u * 72u;
  static_assert(expectedCandidateCount == 4320u);
  if (tuner.info().candidateCount != expectedCandidateCount)
    throw std::logic_error{
        "matrixMultiplication must expose exactly 4320 tuning configurations"};

  auto const bundle =
      KernelBundle{TiledMatrixMultiplicationKernel{},
                   deviceA,
                   deviceB,
                   deviceC,
                   rows,
                   columns,
                   innerDimension,
                   alpakaTune::markTunable(tilesPerGroupTunable),
                   alpakaTune::markTunable(blockRowsTunable),
                   alpakaTune::markTunable(blockColumnsTunable),
                   alpakaTune::markTunable(kTileTunable),
                   alpakaTune::markTunable(rowsPerThreadTunable),
                   alpakaTune::markTunable(simdWidthTunable)};

  Index runs = 0u;
  while (runs < minimumRuns || !tuner.isTuningComplete()) {
    tuner.enqueue(queue, frameSpec, bundle);
    ++runs;
  }
  onHost::memcpy(queue, hostC, deviceC);
  onHost::wait(queue);

  Index errors = 0u;
  for (Index row = 0u; row < rows; ++row) {
    for (Index column = 0u; column < columns; ++column) {
      Scalar expected = 0.0f;
      for (Index inner = 0u; inner < innerDimension; ++inner)
        expected += hostA[row * innerDimension + inner] *
                    hostB[inner * columns + column];
      auto const actual = hostC[row * columns + column];
      auto const tolerance =
          Scalar{2.0e-4f} * std::max(Scalar{1.0f}, std::abs(expected));
      if (std::abs(actual - expected) > tolerance && ++errors <= 8u)
        std::cerr << "C[" << row << "," << column << "] = " << actual
                  << ", expected " << expected << '\n';
    }
  }
  if (errors != 0u) {
    std::cerr << "Matrix multiplication validation failed for " << errors
              << " elements\n";
    return EXIT_FAILURE;
  }

  auto const operations = 2.0 * static_cast<double>(rows) *
                          static_cast<double>(columns) *
                          static_cast<double>(innerDimension);
  std::cout << "Validated " << rows << 'x' << innerDimension << " times "
            << innerDimension << 'x' << columns << " matrix multiplication ("
            << operations / 1.0e9 << " GFLOP per launch, "
            << tuner.info().candidateCount << " candidates)\n";
  return EXIT_SUCCESS;
}

auto parsePositive(char const *text, char const *option) -> Index {
  auto const value = std::stoull(text, nullptr, 0);
  if (value == 0u || value > std::numeric_limits<Index>::max())
    throw std::out_of_range{std::string{option} + " must be a positive size_t"};
  return static_cast<Index>(value);
}

void help(char const *executable) {
  std::cerr
      << executable
      << " [-m rows] [-n columns] [-k inner-dimension] [-r minimum-runs]\n";
}
} // namespace alpaka::example::matrixMultiplication

auto main(int argc, char *argv[]) -> int {
  if (!alpakaTune::consumeBackendOptions(argc, argv))
    return EXIT_FAILURE;

  using namespace alpaka;
  using namespace alpaka::example::matrixMultiplication;
  Index rows = 256u;
  Index columns = 256u;
  Index innerDimension = 256u;
  Index minimumRuns = 1u;

  try {
    int option;
    while ((option = getopt(argc, argv, "hm:n:k:r:")) != -1) {
      switch (option) {
      case 'm':
        rows = parsePositive(optarg, "rows");
        break;
      case 'n':
        columns = parsePositive(optarg, "columns");
        break;
      case 'k':
        innerDimension = parsePositive(optarg, "inner dimension");
        break;
      case 'r':
        minimumRuns = parsePositive(optarg, "minimum runs");
        break;
      case 'h':
        help(argv[0]);
        return EXIT_SUCCESS;
      default:
        help(argv[0]);
        return EXIT_FAILURE;
      }
    }
  } catch (std::exception const &exception) {
    std::cerr << "Invalid argument: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }

  return onHost::executeForEach(
      [=](concepts::BackendSpec auto const &backend) {
        if (!alpakaTune::backendSelected(backend))
          return EXIT_SUCCESS;
        auto selector = onHost::makeDeviceSelector(onHost::DeviceSpec{backend});
        if (!selector.isAvailable())
          return EXIT_SUCCESS;
        return alpaka::example::matrixMultiplication::example(
            onHost::DeviceSpec{backend}, getExecutor(backend), rows, columns,
            innerDimension, minimumRuns);
      },
      onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
