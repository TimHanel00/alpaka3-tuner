// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0
#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/rand/distribution/UniformReal.hpp>
#include <alpaka/rand/engine/philox/philox.hpp>
#include <cstdint>

namespace adaptivePi {
using Index = alpaka::Vec<std::size_t, 1u>;
using TileIndex = alpaka::Vec<std::size_t, 2u>;
inline constexpr std::uint32_t maximumSplitDepth = 8u;
inline constexpr std::uint32_t maximumSamplesPerTile = 2560u;
// Shared reduction storage: next power of two above the 2560-sample limit.
inline constexpr std::uint32_t sampleStorageSize = 4096u;
inline constexpr std::size_t tilesPerAxis = std::size_t{1u}
                                            << maximumSplitDepth;

enum class LeafKind : std::uint32_t { unused, inside, outside, sampled };
// Leaf descriptions support inspection and the independent integration checks.
// They never participate in the numerical reduction or tuning metric.
struct Leaf {
  double area{};
  std::uint32_t depth{}, hits{}, samples{};
  LeafKind kind{LeafKind::unused};
};
struct Tile {
  double x, y, side;
};
ALPAKA_FN_HOST_ACC inline auto tileAt(TileIndex index, std::size_t width)
    -> Tile {
  double const side = 1.0 / width;
  return Tile{index[0u] * side, index[1u] * side, side};
}
ALPAKA_FN_HOST_ACC inline auto contributionIndex(TileIndex index,
                                                 std::size_t width)
    -> TileIndex {
  return index * (tilesPerAxis / width);
}
ALPAKA_FN_HOST_ACC inline auto inside(double x, double y) -> bool {
  return x * x + y * y < 1.0;
}

/** Retire classified tiles; activate four children of each unresolved tile.
 * Masks use fixed positions, so blocks never contend for append counters.
 * A retired leaf writes once at its origin in the contribution buffer. Its
 * descendants remain inactive, preventing double counting at later levels.
 */
template <bool FinalLevel> struct ProcessTiles {
  ALPAKA_FN_ACC void operator()(alpaka::onAcc::concepts::Acc auto const &acc,
                                alpaka::concepts::IMdSpan auto active,
                                alpaka::concepts::IMdSpan auto next,
                                alpaka::concepts::IMdSpan auto contributions,
                                alpaka::concepts::IMdSpan auto leaves,
                                std::size_t width, std::uint32_t depth) const {
    for (auto index :
         alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid,
                                   alpaka::IdxRange{TileIndex{width, width}})) {
      if (!active[index])
        continue;
      auto const tile = tileAt(index, width);
      auto const output = contributionIndex(index, width);
      double const area = tile.side * tile.side;
      if (!inside(tile.x, tile.y)) {
        leaves[output] = Leaf{area, depth, 0u, 0u, LeafKind::outside};
        continue;
      }
      // In the first quadrant the farthest corner proves full inclusion.
      if (inside(tile.x + tile.side, tile.y + tile.side)) {
        contributions[output] = area;
        leaves[output] = Leaf{area, depth, 0u, 0u, LeafKind::inside};
        continue;
      }
      if constexpr (FinalLevel) {
        next[index] = 1u; // Boundary tile: the sampling kernel handles it.
      } else {
        auto const child = index * 2u;
        next[child] = 1u;
        next[child + TileIndex{0u, 1u}] = 1u;
        next[child + TileIndex{1u, 0u}] = 1u;
        next[child + TileIndex{1u, 1u}] = 1u;
      }
    }
  }
};
using RefineTiles = ProcessTiles<false>;
using SelectBoundaryTiles = ProcessTiles<true>;

/** Blocks handle tiles; their threads sample and reduce hits in shared memory.
 */
struct SampleTiles {
  ALPAKA_FN_ACC void operator()(alpaka::onAcc::concepts::Acc auto const &acc,
                                alpaka::concepts::IMdSpan auto boundary,
                                alpaka::concepts::IMdSpan auto contributions,
                                alpaka::concepts::IMdSpan auto leaves,
                                std::size_t width, std::uint32_t depth,
                                std::uint32_t points) const {
    auto hits =
        alpaka::onAcc::declareSharedMdArray<std::uint32_t, alpaka::uniqueId()>(
            acc, alpaka::CVec<std::uint32_t, sampleStorageSize>{});
    for (auto index :
         alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::blocksInGrid,
                                   alpaka::IdxRange{TileIndex{width, width}})) {
      if (!boundary[index])
        continue; // Uniform for the whole block; every worker sees this tile.
      auto const tile = tileAt(index, width);
      std::uint32_t reductionSize = 1u;
      while (reductionSize < points)
        reductionSize *= 2u;
      for (auto sample : alpaka::onAcc::makeIdxMap(
               acc, alpaka::onAcc::worker::linearThreadsInBlock,
               alpaka::IdxRange{Index{reductionSize}}))
        hits[sample.x()] = 0u;
      alpaka::onAcc::syncBlockThreads(acc);

      for (auto sample : alpaka::onAcc::makeIdxMap(
               acc, alpaka::onAcc::worker::linearThreadsInBlock,
               alpaka::IdxRange{Index{points}})) {
        auto const seed = static_cast<std::uint32_t>(
            (index[0u] * width + index[1u]) * maximumSamplesPerTile +
            sample.x());
        auto engine = alpaka::rand::engine::Philox4x32x10{seed};
        auto uniform = alpaka::rand::distribution::UniformReal<double>{};
        double const x = tile.x + tile.side * uniform(engine);
        double const y = tile.y + tile.side * uniform(engine);
        hits[sample.x()] = inside(x, y) ? 1u : 0u;
      }
      alpaka::onAcc::syncBlockThreads(acc);
      // Zero padding supports sample counts that are not powers of two.
      for (auto stride = reductionSize / 2u; stride != 0u; stride /= 2u) {
        for (auto sample : alpaka::onAcc::makeIdxMap(
                 acc, alpaka::onAcc::worker::linearThreadsInBlock,
                 alpaka::IdxRange{Index{stride}}))
          hits[sample.x()] += hits[sample.x() + stride];
        alpaka::onAcc::syncBlockThreads(acc);
      }
      for ([[maybe_unused]] auto writer : alpaka::onAcc::makeIdxMap(
               acc, alpaka::onAcc::worker::linearThreadsInBlock,
               alpaka::IdxRange{Index{1u}})) {
        auto const output = contributionIndex(index, width);
        double const area = tile.side * tile.side;
        contributions[output] = area * hits[0u] / points;
        leaves[output] = Leaf{area, depth, hits[0u], points, LeafKind::sampled};
      }
      // A block may process another tile. Finish all reads before reusing hits.
      alpaka::onAcc::syncBlockThreads(acc);
    }
  }
};
} // namespace adaptivePi
