// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0
#pragma once

// The baseline comes directly from the configured alpaka3 source tree.
#include <StencilKernel.hpp>
#include <alpaka/alpaka.hpp>

#include <array>
#include <cstdint>
#include <utility>

namespace alpakaTune::benchmarks::heatEquation {
using Index = std::uint32_t;
using Vec = alpaka::Vec<Index, 2u>;
enum class Implementation : Index { original, shared, direct, registers };

struct Layout {
  Implementation implementation;
  Index threadsX, threadsY, rows, padding;
  constexpr auto tileX() const -> Index { return threadsX; }
  constexpr auto tileY() const -> Index { return threadsY * rows; }
  constexpr auto workers() const -> Index { return threadsX * threadsY; }
  constexpr auto sharedBytes() const -> Index {
    return implementation == Implementation::shared ||
                   implementation == Implementation::original
               ? (tileY() + 2u) * (tileX() + 2u + padding) * sizeof(double)
               : 0u;
  }
};

inline constexpr auto layouts = [] {
#if ALPAKA_TUNE_HEAT_CPU_CATALOG
  std::array<Layout, 49u> result{};
  Index next{};
  result[next++] = {Implementation::original, 16, 16, 1, 0};
  for (auto x : {64u, 128u, 256u, 512u})
    for (auto y : {1u, 4u, 16u, 64u})
      result[next++] = {Implementation::direct, x, y, 1, 0};
  for (auto x : {32u, 64u, 128u, 256u})
    for (auto y : {4u, 8u, 16u})
      for (auto padding : {0u, 1u})
        result[next++] = {Implementation::shared, x, y, 1, padding};
  for (auto x : {64u, 128u})
    for (auto y : {1u, 4u})
      for (auto rows : {2u, 4u})
        result[next++] = {Implementation::registers, x, y, rows, 0};
#else
  std::array<Layout, 91u> result{};
  Index next{};
  result[next++] = {Implementation::original, 16, 16, 1, 0};
  for (auto x : {32u, 64u, 128u})
    for (auto y : {2u, 4u, 8u})
      for (auto rows : {1u, 2u, 4u})
        for (auto padding : {0u, 1u})
          result[next++] = {Implementation::shared, x, x == 128u ? y / 2u : y,
                            rows, padding};
  for (auto x : {32u, 64u, 128u})
    for (auto y : {1u, 2u})
      for (auto rows : {1u, 2u, 4u})
        result[next++] = {Implementation::direct, x, y, rows, 0};
  for (auto x : {32u, 64u, 128u})
    for (auto y : {1u, 2u})
      for (auto rows : {2u, 4u, 8u})
        result[next++] = {Implementation::registers, x, y, rows, 0};
#endif
  return result;
}();
#if !ALPAKA_TUNE_HEAT_CPU_CATALOG
static_assert(layouts.back().rows == 8u);
static_assert([] {
  for (auto layout : layouts)
    if (layout.workers() > 512u || layout.sharedBytes() >= 19u * 1024u)
      return false;
  return true;
}());
#endif
inline constexpr Index catalogVersion = ALPAKA_TUNE_HEAT_CPU_CATALOG ? 2u : 1u;
inline constexpr Index lastVariant = static_cast<Index>(layouts.size() - 1u);

struct Parameters {
  Index size;
  double dx, dy, dt;
};
struct Choice {
  Index variant, gridPolicy;
};

template <Index Count, typename Function, std::size_t... I>
ALPAKA_FN_ACC inline void unroll(Function const &function,
                                 std::index_sequence<I...>) {
  (function(std::integral_constant<Index, I>{}), ...);
}

struct TunedStencil {
  template <typename Acc, typename Variant>
  ALPAKA_FN_ACC void operator()(Acc const &acc,
                                alpaka::concepts::IMdSpan auto input,
                                alpaka::concepts::IMdSpan auto output,
                                Parameters parameters, Variant) const {
    using namespace alpaka;
    static constexpr auto layout = layouts[Variant::value];
    if constexpr (layout.implementation == Implementation::original) {
      ::StencilKernel{}(acc, input, output, CVec<Index, 16u, 16u>{},
                        CVec<Index, 18u, 18u>{},
                        Vec{parameters.size, parameters.size}, parameters.dx,
                        parameters.dy, parameters.dt);
    } else {
      auto const tileColumns = divCeil(parameters.size, layout.tileX());
      auto const tileRows = divCeil(parameters.size, layout.tileY());
      double const rX = parameters.dt / (parameters.dx * parameters.dx);
      double const rY = parameters.dt / (parameters.dy * parameters.dy);
      double const centerFactor = 1.0 - 2.0 * rX - 2.0 * rY;
      for (auto [tile] :
           onAcc::makeIdxMap(acc, onAcc::worker::linearBlocksInGrid,
                             IdxRange{tileRows * tileColumns})) {
        auto const baseY = (tile / tileColumns) * layout.tileY();
        auto const baseX = (tile % tileColumns) * layout.tileX();
        if constexpr (layout.implementation == Implementation::shared) {
          auto shared = onAcc::declareSharedMdArray<double, uniqueId()>(
              acc, CVec<Index, layout.tileY() + 2u,
                        layout.tileX() + 2u + layout.padding>{});
          // Every worker finishes reading the previous tile before reuse.
          onAcc::syncBlockThreads(acc);
          for (auto [element] : onAcc::makeIdxMap(
                   acc, onAcc::worker::linearThreadsInBlock,
                   IdxRange{(layout.tileY() + 2u) * (layout.tileX() + 2u)})) {
            auto const y = element / (layout.tileX() + 2u);
            auto const x = element % (layout.tileX() + 2u);
            shared[Vec{y, x}] = baseY + y < parameters.size + 2u &&
                                        baseX + x < parameters.size + 2u
                                    ? input[Vec{baseY + y, baseX + x}]
                                    : 0.0;
          }
          onAcc::syncBlockThreads(acc);
          for (auto [worker] :
               onAcc::makeIdxMap(acc, onAcc::worker::linearThreadsInBlock,
                                 IdxRange{layout.workers()})) {
            auto const x = worker % layout.threadsX + 1u;
            auto const workerY = worker / layout.threadsX;
            unroll<layout.rows>(
                [&](auto row) {
                  auto const y = workerY * layout.rows + row + 1u;
                  if (baseY + y <= parameters.size &&
                      baseX + x <= parameters.size)
                    output[Vec{baseY + y, baseX + x}] =
                        shared[Vec{y, x}] * centerFactor +
                        shared[Vec{y, x - 1u}] * rX +
                        shared[Vec{y, x + 1u}] * rX +
                        shared[Vec{y - 1u, x}] * rY +
                        shared[Vec{y + 1u, x}] * rY;
                },
                std::make_index_sequence<layout.rows>{});
          }
        } else {
          for (auto [worker] :
               onAcc::makeIdxMap(acc, onAcc::worker::linearThreadsInBlock,
                                 IdxRange{layout.workers()})) {
            auto const x = baseX + worker % layout.threadsX + 1u;
            auto const y = baseY + worker / layout.threadsX * layout.rows + 1u;
            if (x > parameters.size || y > parameters.size)
              continue;
            if constexpr (layout.implementation == Implementation::direct) {
              unroll<layout.rows>(
                  [&](auto row) {
                    auto const currentY = y + row;
                    if (currentY <= parameters.size)
                      output[Vec{currentY, x}] =
                          input[Vec{currentY, x}] * centerFactor +
                          input[Vec{currentY, x - 1u}] * rX +
                          input[Vec{currentY, x + 1u}] * rX +
                          input[Vec{currentY - 1u, x}] * rY +
                          input[Vec{currentY + 1u, x}] * rY;
                  },
                  std::make_index_sequence<layout.rows>{});
            } else {
              double previous = input[Vec{y - 1u, x}];
              double current = input[Vec{y, x}];
              unroll<layout.rows>(
                  [&](auto row) {
                    auto const currentY = y + row;
                    if (currentY <= parameters.size) {
                      double const next = input[Vec{currentY + 1u, x}];
                      output[Vec{currentY, x}] =
                          current * centerFactor +
                          input[Vec{currentY, x - 1u}] * rX +
                          input[Vec{currentY, x + 1u}] * rX + previous * rY +
                          next * rY;
                      previous = current;
                      current = next;
                    }
                  },
                  std::make_index_sequence<layout.rows>{});
            }
          }
        }
      }
    }
  }
};
} // namespace alpakaTune::benchmarks::heatEquation
