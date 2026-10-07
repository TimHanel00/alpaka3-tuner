// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <array>
#include <cstdint>
#include <utility>

namespace alpakaTune::benchmarks::matmul {
using Index = std::uint32_t;

struct Shape {
  Index m, n, k;
};

struct Tile {
  Index m, n, k, rowsPerThread, columnsPerThread, stages;

  constexpr auto sharedBytes() const -> Index {
    return stages * k * (m + n) * sizeof(float);
  }
  constexpr auto blocks(Shape shape) const -> Index {
    return alpaka::divCeil(shape.m, m) * alpaka::divCeil(shape.n, n);
  }
};

// Each layout has exactly 256 logical workers. Coupled tile/register choices
// avoid the mostly invalid Cartesian product of independently chosen sizes.
inline constexpr std::array tiles{
    Tile{64, 64, 16, 4, 4, 1},   Tile{64, 64, 16, 4, 4, 2},
    Tile{64, 64, 32, 4, 4, 1},   Tile{64, 128, 16, 4, 8, 1},
    Tile{64, 128, 16, 4, 8, 2},  Tile{64, 128, 32, 4, 8, 1},
    Tile{128, 64, 16, 8, 4, 1},  Tile{128, 64, 16, 8, 4, 2},
    Tile{128, 64, 32, 8, 4, 1},  Tile{128, 128, 16, 8, 8, 1},
    Tile{128, 128, 16, 8, 8, 2}, Tile{128, 128, 32, 8, 8, 1}};

static_assert([] {
  for (auto tile : tiles)
    if (tile.m * tile.n / (tile.rowsPerThread * tile.columnsPerThread) !=
            256u ||
        tile.sharedBytes() > 32u * 1024u)
      return false;
  return true;
}());

// A pack expansion keeps accumulator indices compile-time constants, including
// with host compilers. This avoids spilling a dynamically indexed register
// tile.
template <typename Function, std::size_t... Indices>
ALPAKA_FN_ACC inline void staticForImpl(Function const &function,
                                        std::index_sequence<Indices...>) {
  (function(std::integral_constant<Index, Indices>{}), ...);
}

template <Index Count, typename Function>
ALPAKA_FN_ACC inline void staticFor(Function const &function) {
  staticForImpl(function, std::make_index_sequence<Count>{});
}

struct MatmulKernel {
  template <typename Acc, typename Variant>
  ALPAKA_FN_ACC void
  operator()(Acc const &acc, alpaka::concepts::IMdSpan auto a,
             alpaka::concepts::IMdSpan auto b, alpaka::concepts::IMdSpan auto c,
             Shape shape, bool vectorLoads, Variant) const {
    static constexpr auto tile = tiles[Variant::value];
    constexpr auto threadRows = tile.m / tile.rowsPerThread;
    constexpr auto threadColumns = tile.n / tile.columnsPerThread;
    constexpr auto logicalThreads = threadRows * threadColumns;
    auto sharedA =
        alpaka::onAcc::declareSharedMdArray<float, alpaka::uniqueId()>(
            acc, alpaka::CVec<Index, tile.stages * tile.k, tile.m>{});
    auto sharedB =
        alpaka::onAcc::declareSharedMdArray<float, alpaka::uniqueId()>(
            acc, alpaka::CVec<Index, tile.stages * tile.k, tile.n>{});
    auto const thread = acc.getIdxWithin(alpaka::onAcc::origin::block,
                                         alpaka::onAcc::unit::threads)
                            .product();
    // A one-dimensional index's product equals its only component.
    auto const threads = acc.getExtentsOf(alpaka::onAcc::origin::block,
                                          alpaka::onAcc::unit::threads)
                             .product();
    auto const tileColumns = alpaka::divCeil(shape.n, tile.n);

    for (auto [block] :
         alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::blocksInGrid,
                                   alpaka::IdxRange{tile.blocks(shape)})) {
      auto const rowBase = (block / tileColumns) * tile.m;
      auto const columnBase = (block % tileColumns) * tile.n;
      // Serial/blocks-only executors emulate the logical workers in rounds;
      // GPU executors perform exactly one round with 256 physical threads.
      for (Index round = 0; round < logicalThreads; round += threads) {
        auto const worker = round + thread;
        auto const active = worker < logicalThreads;
        auto const workerRow = worker / threadColumns;
        auto const workerColumn = worker % threadColumns;
        float accumulators[tile.rowsPerThread][tile.columnsPerThread]{};

        auto load = [&](Index kBase, Index stage) {
          for (Index pack = thread; pack < tile.m * tile.k / 4u;
               pack += threads) {
            auto const row = pack / (tile.k / 4u);
            auto const inner = (pack % (tile.k / 4u)) * 4u;
            if (vectorLoads && rowBase + row < shape.m &&
                kBase + inner + 3u < shape.k) {
              auto const values =
                  alpaka::SimdPtr{a, alpaka::Vec{rowBase + row, kBase + inner},
                                  alpaka::Alignment<16>{},
                                  alpaka::CVec<Index, 4>{}}
                      .load();
              staticFor<4>([&](auto lane) {
                sharedA[alpaka::Vec{stage * tile.k + inner + lane, row}] =
                    values[Index{lane}];
              });
            } else {
              staticFor<4>([&](auto lane) {
                sharedA[alpaka::Vec{stage * tile.k + inner + lane, row}] =
                    rowBase + row < shape.m && kBase + inner + lane < shape.k
                        ? a[alpaka::Vec{rowBase + row, kBase + inner + lane}]
                        : 0.0f;
              });
            }
          }
          for (Index pack = thread; pack < tile.k * tile.n / 4u;
               pack += threads) {
            auto const inner = pack / (tile.n / 4u);
            auto const column = (pack % (tile.n / 4u)) * 4u;
            if (vectorLoads && kBase + inner < shape.k &&
                columnBase + column + 3u < shape.n) {
              auto const values =
                  alpaka::SimdPtr{
                      b, alpaka::Vec{kBase + inner, columnBase + column},
                      alpaka::Alignment<16>{}, alpaka::CVec<Index, 4>{}}
                      .load();
              staticFor<4>([&](auto lane) {
                sharedB[alpaka::Vec{stage * tile.k + inner, column + lane}] =
                    values[Index{lane}];
              });
            } else {
              staticFor<4>([&](auto lane) {
                sharedB[alpaka::Vec{stage * tile.k + inner, column + lane}] =
                    kBase + inner < shape.k &&
                            columnBase + column + lane < shape.n
                        ? b[alpaka::Vec{kBase + inner,
                                        columnBase + column + lane}]
                        : 0.0f;
              });
            }
          }
        };

        if constexpr (tile.stages == 2u) {
          load(0u, 0u);
          alpaka::onAcc::syncBlockThreads(acc);
        }
        for (Index kBase = 0u, iteration = 0u; kBase < shape.k;
             kBase += tile.k, ++iteration) {
          auto const stage = iteration % tile.stages;
          if constexpr (tile.stages == 1u) {
            load(kBase, stage);
            alpaka::onAcc::syncBlockThreads(acc);
          } else if (kBase + tile.k < shape.k) {
            // Separate shared buffers allow loading the next tile before
            // consuming the current one, with one barrier per K tile.
            load(kBase + tile.k, 1u - stage);
          }
          if (active) {
            staticFor<tile.k>([&](auto inner) {
              float valuesA[tile.rowsPerThread];
              float valuesB[tile.columnsPerThread];
              staticFor<tile.rowsPerThread>([&](auto row) {
                valuesA[row] =
                    sharedA[alpaka::Vec{stage * tile.k + Index{inner},
                                        workerRow + row * threadRows}];
              });
              staticFor<tile.columnsPerThread>([&](auto column) {
                valuesB[column] =
                    sharedB[alpaka::Vec{stage * tile.k + Index{inner},
                                        workerColumn + column * threadColumns}];
              });
              staticFor<tile.rowsPerThread>([&](auto row) {
                staticFor<tile.columnsPerThread>([&](auto column) {
                  accumulators[row][column] = alpaka::math::fma(
                      valuesA[row], valuesB[column], accumulators[row][column]);
                });
              });
            });
          }
          alpaka::onAcc::syncBlockThreads(acc);
        }
        if (active) {
          staticFor<tile.rowsPerThread>([&](auto row) {
            auto const globalRow = rowBase + workerRow + row * threadRows;
            staticFor<tile.columnsPerThread>([&](auto column) {
              auto const globalColumn =
                  columnBase + workerColumn + column * threadColumns;
              if (globalRow < shape.m && globalColumn < shape.n)
                c[alpaka::Vec{globalRow, globalColumn}] =
                    accumulators[row][column];
            });
          });
        }
      }
    }
  }
};

} // namespace alpakaTune::benchmarks::matmul
