// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0
#pragma once
#include <array>
#include <cstdint>
#include <string_view>
namespace alpakaTune::benchmarks::heatEquation {
struct SelectedWinner {
  std::uint32_t size, variant, gridPolicy;
};
inline constexpr std::string_view selectedDevice =
    "AMD EPYC 9654 96-Core Processor                ";
inline constexpr std::string_view selectedArchitecture = "x86_64";
inline constexpr std::string_view selectedExecutor =
    "alpaka::exec::CpuOmpBlocks";
inline constexpr std::uint32_t selectedCatalog = 2,
                               selectedMultiProcessors = 192,
                               selectedOmpThreads = 192;
inline constexpr std::array selectedWinners{
    SelectedWinner{128, 4, 0},
    SelectedWinner{256, 4, 0},
    SelectedWinner{512, 48, 0},
    // Keep the original: it was faster in the complete-step comparison.
    SelectedWinner{1024, 0, 0},
    SelectedWinner{2048, 9, 1},
    SelectedWinner{4096, 46, 1},
    SelectedWinner{8192, 5, 0},
    SelectedWinner{16384, 1, 1},
};
} // namespace alpakaTune::benchmarks::heatEquation
