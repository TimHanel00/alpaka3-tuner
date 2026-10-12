// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <array>
#include <cstdint>
#include <utility>

namespace alpakaTune::example::matmul {
using Index = std::uint32_t;

struct Shape {
  Index m, n, k;
};

struct Tile {
  Index m, n, k, rowsPerThread, columnsPerThread, stages;
  Index padding{};
  bool warpLayout{}, asynchronous{}, registerPrefetch{};
  bool interleavedRegisters{};
  Index sharedSwizzle{}; // Bit 0 permutes B; bit 1 permutes A.

  constexpr auto sharedBytes() const -> Index {
    return stages * k * (m + padding + n) * sizeof(float);
  }
  constexpr auto blocks(Shape shape) const -> Index {
    return alpaka::divCeil(shape.m, m) * alpaka::divCeil(shape.n, n);
  }
  constexpr auto logicalThreads() const -> Index {
    return (m / rowsPerThread) * (n / columnsPerThread);
  }
};

// The original baseline and the layouts selected by the GPU/CPU campaigns.
// Only these layouts are instantiated by the shipped winner catalog.
inline constexpr std::array tiles{
    Tile{64, 64, 16, 4, 4, 1}, Tile{32, 64, 32, 4, 4, 1, 4, true, false, true},
    Tile{32, 64, 32, 4, 4, 2, 4, true, true, true},
    Tile{64, 128, 32, 8, 8, 1, 4, true, false, true},
    Tile{32, 64, 16, 4, 4, 1, 4, true, false, true}};

static_assert([] {
  for (auto tile : tiles)
    if (tile.m % tile.rowsPerThread || tile.n % tile.columnsPerThread ||
        (tile.logicalThreads() != 128u && tile.logicalThreads() != 256u &&
         tile.logicalThreads() != 512u) ||
        tile.sharedBytes() > 48u * 1024u ||
        (tile.asynchronous && tile.stages != 2u) || tile.sharedSwizzle > 3u ||
        (tile.sharedSwizzle &&
         (!tile.warpLayout || tile.interleavedRegisters ||
          tile.rowsPerThread != 8u || tile.columnsPerThread != 8u)) ||
        (tile.interleavedRegisters &&
         (!tile.warpLayout || tile.rowsPerThread < 4u ||
          tile.columnsPerThread < 4u)) ||
        (tile.warpLayout &&
         (tile.m % (4u * tile.rowsPerThread) ||
          tile.n % (8u * tile.columnsPerThread) || tile.padding % 4u ||
          (tile.rowsPerThread != 2u && tile.rowsPerThread % 4u) ||
          (tile.columnsPerThread != 2u && tile.columnsPerThread % 4u))))
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

struct LegacyMatmulKernel {
  template <typename Acc, typename Variant>
  ALPAKA_FN_ACC void
  operator()(Acc const &acc, alpaka::concepts::IMdSpan auto a,
             alpaka::concepts::IMdSpan auto b, alpaka::concepts::IMdSpan auto c,
             Shape inputShape, bool vectorLoads, Variant) const {
    auto const shape = [&] {
      if constexpr (requires { Variant::shape; })
        return Variant::shape;
      else
        return inputShape;
    }();
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
    auto const threads = [&] {
      if constexpr (requires { Variant::numThreads; })
        return Variant::numThreads;
      else
        return acc
            .getExtentsOf(alpaka::onAcc::origin::block,
                          alpaka::onAcc::unit::threads)
            .product();
    }();
    auto const gridStride = [&] {
      if constexpr (requires { Variant::numBlocks; })
        return Variant::numBlocks;
      else
        return acc
            .getExtentsOf(alpaka::onAcc::origin::grid,
                          alpaka::onAcc::unit::blocks)
            .product();
    }();
    auto const tileColumns = alpaka::divCeil(shape.n, tile.n);

    for (Index block = acc.getIdxWithin(alpaka::onAcc::origin::grid,
                                        alpaka::onAcc::unit::blocks)
                           .product();
         block < tile.blocks(shape); block += gridStride) {
      auto const rowBase = (block / tileColumns) * tile.m;
      auto const columnBase = (block % tileColumns) * tile.n;
      // Serial/blocks-only executors emulate the logical workers in rounds;
      // GPU executors perform one round with the layout's physical threads.
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

namespace internal {
// The CUDA customization is confined to data movement. Other accelerators use
// the same layout and pipeline with synchronous copies.
template <Index Bytes>
ALPAKA_FN_ACC inline void copySharedAsync(float *destination,
                                          float const *source, bool valid) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  auto const sharedAddress =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(destination));
  auto const sourceBytes = valid ? Bytes : 0u;
  asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;"
               :
               : "r"(sharedAddress), "l"(source), "n"(Bytes), "r"(sourceBytes)
               : "memory");
#else
  staticFor<Bytes / sizeof(float)>([&](auto lane) {
    destination[Index{lane}] = valid ? source[Index{lane}] : 0.0f;
  });
#endif
}

ALPAKA_FN_ACC inline void commitCopies() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;" : : : "memory");
#endif
}
ALPAKA_FN_ACC inline void waitCopies() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;" : : : "memory");
#endif
}

template <Index Width>
ALPAKA_FN_ACC inline void loadShared(float const *source,
                                     float (&values)[Width]) {
  static_assert(Width == 2u || Width == 4u);
#if defined(__CUDA_ARCH__)
  auto const address =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(source));
  if constexpr (Width == 4u)
    asm("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(values[0]), "=f"(values[1]), "=f"(values[2]), "=f"(values[3])
        : "r"(address)
        : "memory");
  else
    asm("ld.shared.v2.f32 {%0, %1}, [%2];"
        : "=f"(values[0]), "=f"(values[1])
        : "r"(address)
        : "memory");
#else
  staticFor<Width>(
      [&](auto lane) { values[Index{lane}] = source[Index{lane}]; });
#endif
}

// Permute four-float groups within one warp extent, preserving vector alignment
// and each thread's logical output coordinates.
template <Index Groups>
ALPAKA_FN_ACC constexpr auto sharedCoordinate(Index coordinate) -> Index {
  constexpr auto extent = Groups * 8u;
  return coordinate / extent * extent + coordinate % 4u +
         (coordinate / 4u % 2u) * Groups * 4u + (coordinate / 8u % Groups) * 4u;
}

template <Index Rows, Index Columns> struct alignas(16) SharedTile {
  float values[Rows][Columns];
};
} // namespace internal

struct WarpMatmulKernel {
  template <typename Acc, typename Configuration>
  ALPAKA_FN_ACC void
  operator()(Acc const &acc, alpaka::concepts::IMdSpan auto a,
             alpaka::concepts::IMdSpan auto b, alpaka::concepts::IMdSpan auto c,
             Shape inputShape, bool vectorLoads, Configuration) const {
    static constexpr auto tile = tiles[Configuration::value];
    auto const shape = [&] {
      if constexpr (requires { Configuration::shape; })
        return Configuration::shape;
      else
        return inputShape;
    }();
    constexpr bool fullTiles = [] {
      if constexpr (requires { Configuration::shape; }) {
        constexpr auto s = Configuration::shape;
        constexpr auto layout = tiles[Configuration::value];
        return s.m % layout.m == 0u && s.n % layout.n == 0u &&
               s.k % layout.k == 0u;
      } else
        return false;
    }();
    auto &sharedA = alpaka::onAcc::declareSharedVar<
        internal::SharedTile<tile.stages * tile.k, tile.m + tile.padding>,
        alpaka::uniqueId()>(acc);
    auto &sharedB = alpaka::onAcc::declareSharedVar<
        internal::SharedTile<tile.stages * tile.k, tile.n>, alpaka::uniqueId()>(
        acc);
    auto const thread = acc.getIdxWithin(alpaka::onAcc::origin::block,
                                         alpaka::onAcc::unit::threads)
                            .product();
    auto const threads = [&] {
      if constexpr (requires { Configuration::numThreads; })
        return Configuration::numThreads;
      else
        return acc
            .getExtentsOf(alpaka::onAcc::origin::block,
                          alpaka::onAcc::unit::threads)
            .product();
    }();
    auto const gridStride = [&] {
      if constexpr (requires { Configuration::numBlocks; })
        return Configuration::numBlocks;
      else
        return acc
            .getExtentsOf(alpaka::onAcc::origin::grid,
                          alpaka::onAcc::unit::blocks)
            .product();
    }();
    constexpr auto warpColumns = tile.n / (8u * tile.columnsPerThread);
    auto sharedRow = [](Index row) {
      if constexpr (tile.sharedSwizzle & 2u)
        return internal::sharedCoordinate<4u>(row);
      else
        return row;
    };
    auto sharedColumn = [](Index column) {
      if constexpr (tile.sharedSwizzle & 1u)
        return internal::sharedCoordinate<8u>(column);
      else
        return column;
    };
    auto const tileColumns = alpaka::divCeil(shape.n, tile.n);
    for (Index block = acc.getIdxWithin(alpaka::onAcc::origin::grid,
                                        alpaka::onAcc::unit::blocks)
                           .product();
         block < tile.blocks(shape); block += gridStride) {
      auto const rowBase = (block / tileColumns) * tile.m;
      auto const columnBase = (block % tileColumns) * tile.n;
      for (Index round = 0u; round < tile.logicalThreads(); round += threads) {
        auto const worker = round + thread;
        auto const warp = worker / 32u;
        auto const lane = worker % 32u;
        constexpr auto laneRows =
            tile.interleavedRegisters ? 4u : tile.rowsPerThread;
        constexpr auto laneColumns =
            tile.interleavedRegisters ? 4u : tile.columnsPerThread;
        auto const workerRow = (warp / warpColumns) * 4u * tile.rowsPerThread +
                               (lane / 8u) * laneRows;
        auto const workerColumn =
            (warp % warpColumns) * 8u * tile.columnsPerThread +
            (lane % 8u) * laneColumns;
        auto rowOffset = [](Index row) {
          if constexpr (tile.interleavedRegisters)
            return row % 4u + (row / 4u) * 16u;
          else
            return row;
        };
        auto columnOffset = [](Index column) {
          if constexpr (tile.interleavedRegisters)
            return column % 4u + (column / 4u) * 32u;
          else
            return column;
        };
        float accumulators[tile.rowsPerThread][tile.columnsPerThread]{};

        auto load = [&](Index kBase, Index stage) {
          for (Index pack = thread; pack < tile.m * tile.k / 4u;
               pack += threads) {
            auto const row = pack / (tile.k / 4u);
            auto const inner = (pack % (tile.k / 4u)) * 4u;
            if constexpr (tile.asynchronous) {
              staticFor<4>([&](auto component) {
                auto const valid =
                    fullTiles || (rowBase + row < shape.m &&
                                  kBase + inner + component < shape.k);
                auto const *source =
                    valid ? &a[alpaka::Vec{rowBase + row,
                                           kBase + inner + Index{component}}]
                          : &a[alpaka::Vec{0u, 0u}];
                internal::copySharedAsync<4>(
                    &sharedA.values[stage * tile.k + inner + Index{component}]
                                   [sharedRow(row)],
                    source, valid);
              });
            } else if (vectorLoads &&
                       (fullTiles || (rowBase + row < shape.m &&
                                      kBase + inner + 3u < shape.k))) {
              auto const values =
                  alpaka::SimdPtr{a, alpaka::Vec{rowBase + row, kBase + inner},
                                  alpaka::Alignment<16>{},
                                  alpaka::CVec<Index, 4>{}}
                      .load();
              staticFor<4>([&](auto component) {
                sharedA.values[stage * tile.k + inner + Index{component}]
                              [sharedRow(row)] = values[Index{component}];
              });
            } else {
              staticFor<4>([&](auto component) {
                sharedA.values[stage * tile.k + inner + Index{component}]
                              [sharedRow(row)] =
                    rowBase + row < shape.m &&
                            kBase + inner + component < shape.k
                        ? a[alpaka::Vec{rowBase + row,
                                        kBase + inner + Index{component}}]
                        : 0.0f;
              });
            }
          }
          for (Index pack = thread; pack < tile.k * tile.n / 4u;
               pack += threads) {
            auto const inner = pack / (tile.n / 4u);
            auto const column = (pack % (tile.n / 4u)) * 4u;
            auto const validPack =
                fullTiles ||
                (kBase + inner < shape.k && columnBase + column + 3u < shape.n);
            if (vectorLoads && validPack) {
              if constexpr (tile.asynchronous)
                internal::copySharedAsync<16>(
                    &sharedB
                         .values[stage * tile.k + inner][sharedColumn(column)],
                    &b[alpaka::Vec{kBase + inner, columnBase + column}], true);
              else {
                auto const values =
                    alpaka::SimdPtr{
                        b, alpaka::Vec{kBase + inner, columnBase + column},
                        alpaka::Alignment<16>{}, alpaka::CVec<Index, 4>{}}
                        .load();
                values.copyTo(&sharedB.values[stage * tile.k + inner]
                                             [sharedColumn(column)],
                              alpaka::Alignment<16>{});
              }
            } else {
              staticFor<4>([&](auto component) {
                auto const valid = kBase + inner < shape.k &&
                                   columnBase + column + component < shape.n;
                if constexpr (tile.asynchronous) {
                  auto const *source =
                      valid
                          ? &b[alpaka::Vec{kBase + inner, columnBase + column +
                                                              Index{component}}]
                          : &b[alpaka::Vec{0u, 0u}];
                  internal::copySharedAsync<4>(
                      &sharedB.values[stage * tile.k + inner]
                                     [sharedColumn(column + Index{component})],
                      source, valid);
                } else
                  sharedB.values[stage * tile.k + inner]
                                [sharedColumn(column + Index{component})] =
                      valid
                          ? b[alpaka::Vec{kBase + inner, columnBase + column +
                                                             Index{component}}]
                          : 0.0f;
              });
            }
          }
          if constexpr (tile.asynchronous)
            internal::commitCopies();
        };
        load(0u, 0u);
        if constexpr (tile.asynchronous)
          internal::waitCopies();
        alpaka::onAcc::syncBlockThreads(acc);
        for (Index kBase = 0u, iteration = 0u; kBase < shape.k;
             kBase += tile.k, ++iteration) {
          auto const stage = iteration % tile.stages;
          if constexpr (tile.stages == 2u)
            if (kBase + tile.k < shape.k)
              load(kBase + tile.k, 1u - stage);
          constexpr auto fragments = tile.registerPrefetch ? 2u : 1u;
          float valuesA[fragments][tile.rowsPerThread];
          float valuesB[fragments][tile.columnsPerThread];
          constexpr auto rowPack = tile.rowsPerThread < 4u ? 2u : 4u;
          constexpr auto columnPack = tile.columnsPerThread < 4u ? 2u : 4u;
          auto readFragment = [&](auto inner, auto fragment) {
            staticFor<tile.rowsPerThread / rowPack>([&](auto pack) {
              float values[rowPack];
              internal::loadShared<rowPack>(
                  &sharedA.values[stage * tile.k + Index{inner}][sharedRow(
                      workerRow + rowOffset(Index{pack} * rowPack))],
                  values);
              staticFor<rowPack>([&](auto component) {
                valuesA[Index{fragment}]
                       [Index{pack} * rowPack + Index{component}] =
                           values[Index{component}];
              });
            });
            staticFor<tile.columnsPerThread / columnPack>([&](auto pack) {
              float values[columnPack];
              internal::loadShared<columnPack>(
                  &sharedB.values[stage * tile.k + Index{inner}][sharedColumn(
                      workerColumn + columnOffset(Index{pack} * columnPack))],
                  values);
              staticFor<columnPack>([&](auto component) {
                valuesB[Index{fragment}]
                       [Index{pack} * columnPack + Index{component}] =
                           values[Index{component}];
              });
            });
          };
          if constexpr (tile.registerPrefetch)
            readFragment(std::integral_constant<Index, 0u>{},
                         std::integral_constant<Index, 0u>{});
          staticFor<tile.k>([&](auto inner) {
            constexpr auto fragment = Index{inner} % fragments;
            if constexpr (tile.registerPrefetch) {
              if constexpr (Index{inner} + 1u < tile.k)
                readFragment(std::integral_constant<Index, Index{inner} + 1u>{},
                             std::integral_constant<Index, 1u - fragment>{});
            } else
              readFragment(inner, std::integral_constant<Index, 0u>{});
            staticFor<tile.rowsPerThread>([&](auto row) {
              staticFor<tile.columnsPerThread>([&](auto column) {
                accumulators[row][column] = alpaka::math::fma(
                    valuesA[fragment][row], valuesB[fragment][column],
                    accumulators[row][column]);
              });
            });
          });
          if constexpr (tile.asynchronous)
            internal::waitCopies();
          alpaka::onAcc::syncBlockThreads(acc);
          if constexpr (tile.stages == 1u) {
            if (kBase + tile.k < shape.k) {
              load(kBase + tile.k, 0u);
              alpaka::onAcc::syncBlockThreads(acc);
            }
          }
        }
        staticFor<tile.rowsPerThread>([&](auto row) {
          auto const globalRow = rowBase + workerRow + rowOffset(Index{row});
          constexpr auto outputPack = tile.columnsPerThread < 4u ? 2u : 4u;
          staticFor<tile.columnsPerThread / outputPack>([&](auto pack) {
            auto const globalColumn = columnBase + workerColumn +
                                      columnOffset(Index{pack} * outputPack);
            if (vectorLoads &&
                (fullTiles || (globalRow < shape.m &&
                               globalColumn + outputPack - 1u < shape.n))) {
              auto values = alpaka::Simd<float, outputPack>{};
              staticFor<outputPack>([&](auto component) {
                values[Index{component}] =
                    accumulators[row]
                                [Index{pack} * outputPack + Index{component}];
              });
              values.copyTo(&c[alpaka::Vec{globalRow, globalColumn}],
                            alpaka::Alignment<outputPack * sizeof(float)>{});
            } else
              staticFor<outputPack>([&](auto component) {
                if (globalRow < shape.m && globalColumn + component < shape.n)
                  c[alpaka::Vec{globalRow, globalColumn + Index{component}}] =
                      accumulators[row]
                                  [Index{pack} * outputPack + Index{component}];
              });
          });
        });
        alpaka::onAcc::syncBlockThreads(acc);
      }
    }
  }
};

struct MatmulKernel {
  template <typename Acc, typename Configuration>
  ALPAKA_FN_ACC void
  operator()(Acc const &acc, alpaka::concepts::IMdSpan auto a,
             alpaka::concepts::IMdSpan auto b, alpaka::concepts::IMdSpan auto c,
             Shape shape, bool vectorLoads, Configuration tag) const {
    if constexpr (tiles[Configuration::value].warpLayout)
      WarpMatmulKernel{}(acc, a, b, c, shape, vectorLoads, tag);
    else
      LegacyMatmulKernel{}(acc, a, b, c, shape, vectorLoads, tag);
  }
};
} // namespace alpakaTune::example::matmul
