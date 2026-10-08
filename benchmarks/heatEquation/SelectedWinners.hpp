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
inline constexpr std::string_view selectedDevice = "NVIDIA A30";
inline constexpr std::string_view selectedArchitecture = "sm_80";
inline constexpr std::string_view selectedExecutor = "alpaka::exec::GpuCuda";
inline constexpr std::uint32_t selectedCatalog = 1,
                               selectedMultiProcessors = 56,
                               selectedOmpThreads = 0;
inline constexpr std::array selectedWinners{
    SelectedWinner{128, 64, 0},  SelectedWinner{256, 70, 0},
    SelectedWinner{512, 88, 0},  SelectedWinner{1024, 72, 2},
    SelectedWinner{2048, 83, 0}, SelectedWinner{4096, 86, 0},
    SelectedWinner{8192, 83, 0}, SelectedWinner{16384, 83, 0},
};
} // namespace alpakaTune::benchmarks::heatEquation
