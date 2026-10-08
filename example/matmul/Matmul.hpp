// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "MatmulKernel.hpp"

#include <algorithm>
#include <concepts>
#include <functional>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#if ALPAKA_OMP
#include <omp.h>
#endif

namespace alpakaTune::example::matmul {
template <Index M, Index N, Index K, Index Layout, Index Threads, Index Blocks>
struct Configuration : std::integral_constant<Index, Layout> {
  static constexpr Shape shape{M, N, K};
  static constexpr Index numThreads = Threads;
  static constexpr Index numBlocks = Blocks;
  static_assert(Layout < tiles.size());
  static_assert(M > 0u && N > 0u && K > 0u);
  static_assert(std::max({M, N, K}) <= (1u << 20u));
  static_assert(Threads == 1u || Threads == tiles[Layout].logicalThreads());
  static_assert(Blocks > 0u && Blocks <= tiles[Layout].blocks(shape));
};

// Exact, independently evaluated winners. Layouts 1/2/3 correspond to the
// campaign's layouts 53/54/61. No losing search candidates are retained.
inline constexpr auto winners =
    std::tuple{Configuration<512, 512, 512, 1, 128, 128>{},
               Configuration<1024, 1024, 1024, 1, 128, 512>{},
               Configuration<2048, 2048, 2048, 1, 128, 2048>{},
               Configuration<4096, 4096, 4096, 3, 128, 2048>{},
               Configuration<256, 2048, 1024, 1, 128, 256>{},
               Configuration<2048, 256, 1024, 2, 128, 256>{},
               Configuration<1023, 1009, 997, 1, 128, 512>{}};

struct CpuContext {
  Index processingUnits{};
  // Runtime team size; the selected configuration independently sets numBlocks.
  Index ompThreads{1u};
};

template <Index ProcessingUnits, Index OmpThreads, typename Selected>
struct CpuConfiguration : Selected {
  static constexpr Index numProcessingUnits = ProcessingUnits;
  static constexpr Index numOmpThreads = OmpThreads;
};

inline constexpr auto cpuWinners = std::tuple{
    CpuConfiguration<128, 128, Configuration<512, 512, 512, 4, 1, 128>>{},
    CpuConfiguration<128, 128, Configuration<1024, 1024, 1024, 3, 1, 128>>{},
    CpuConfiguration<128, 128, Configuration<2048, 2048, 2048, 3, 1, 512>>{},
    CpuConfiguration<128, 128, Configuration<4096, 4096, 4096, 3, 1, 2048>>{},
    CpuConfiguration<128, 128, Configuration<256, 2048, 1024, 0, 1, 128>>{},
    CpuConfiguration<128, 128, Configuration<2048, 256, 1024, 0, 1, 128>>{},
    CpuConfiguration<128, 128, Configuration<1023, 1009, 997, 3, 1, 128>>{}};

namespace internal {
template <typename Function>
bool selectGpuConfiguration(std::string_view device, std::string_view api,
                            std::string_view executor, Shape shape,
                            Function &&function) {
  if (device != "NVIDIA A30" || api != "Cuda" ||
      executor != "alpaka::exec::GpuCuda")
    return false;
  return std::apply(
      [&](auto... configurations) {
        return ((shape.m == decltype(configurations)::shape.m &&
                 shape.n == decltype(configurations)::shape.n &&
                 shape.k == decltype(configurations)::shape.k &&
                 (std::invoke(function, configurations), true)) ||
                ...);
      },
      winners);
}

template <typename Function>
bool selectCpuConfiguration(std::string_view device, std::string_view api,
                            std::string_view executor, Shape shape,
                            CpuContext context, Function &&function) {
  while (!device.empty() && device.back() == ' ')
    device.remove_suffix(1u);
  if (device != "AMD EPYC 7713 64-Core Processor" || api != "Host" ||
      executor != "alpaka::exec::CpuOmpBlocks" ||
      context.processingUnits != 128u)
    return false;
  return std::apply(
      [&](auto... configurations) {
        return (
            (context.ompThreads == decltype(configurations)::numOmpThreads &&
             shape.m == decltype(configurations)::shape.m &&
             shape.n == decltype(configurations)::shape.n &&
             shape.k == decltype(configurations)::shape.k &&
             (std::invoke(function, configurations), true)) ||
            ...);
      },
      cpuWinners);
}
} // namespace internal

// The callback receives a configuration tag whose tile, matrix dimensions,
// thread count and block count are all compile-time constants.
template <typename Function>
bool selectConfiguration(std::string_view device, std::string_view api,
                         std::string_view executor, Shape shape,
                         Function &&function, CpuContext cpu = {}) {
  if (api == "Host")
    return internal::selectCpuConfiguration(device, api, executor, shape, cpu,
                                            std::forward<Function>(function));
  return internal::selectGpuConfiguration(device, api, executor, shape,
                                          std::forward<Function>(function));
}

namespace internal {
template <typename Selected, Index Threads>
struct PhysicalConfiguration : Selected {
  static constexpr Index numThreads = Threads;
};

struct BaselineConfiguration : std::integral_constant<Index, 0u> {
  static constexpr Index numThreads = 256u;
};

inline auto matrixShape(alpaka::concepts::IBuffer auto const &a,
                        alpaka::concepts::IBuffer auto const &b,
                        alpaka::concepts::IBuffer auto const &c) -> Shape {
  static_assert(ALPAKA_TYPEOF(a)::dim() == 2u &&
                ALPAKA_TYPEOF(b)::dim() == 2u && ALPAKA_TYPEOF(c)::dim() == 2u);
  static_assert(std::same_as<typename ALPAKA_TYPEOF(a)::value_type, float> &&
                std::same_as<typename ALPAKA_TYPEOF(b)::value_type, float> &&
                std::same_as<typename ALPAKA_TYPEOF(c)::value_type, float>);
  auto const ae = alpaka::onHost::getExtents(a);
  auto const be = alpaka::onHost::getExtents(b);
  auto const ce = alpaka::onHost::getExtents(c);
  if (a.getPitches()[1u] != sizeof(float) ||
      b.getPitches()[1u] != sizeof(float) ||
      c.getPitches()[1u] != sizeof(float))
    throw std::invalid_argument{
        "Expected row-major matrices with contiguous columns"};
  if (ae[1u] != be[0u] || ce[0u] != ae[0u] || ce[1u] != be[1u] ||
      ae[0u] == 0u || ae[1u] == 0u || be[1u] == 0u ||
      std::max({ae[0u], ae[1u], be[1u]}) > (1u << 20u))
    throw std::invalid_argument{"Expected A[M,K], B[K,N], C[M,N], "
                                "with dimensions in [1,1048576]"};
  return {static_cast<Index>(ae[0u]), static_cast<Index>(be[1u]),
          static_cast<Index>(ae[1u])};
}

inline bool vectorLoads(alpaka::concepts::IBuffer auto const &a,
                        alpaka::concepts::IBuffer auto const &b,
                        alpaka::concepts::IBuffer auto const &c) {
  return a.getPitches()[0u] % 16u == 0u && b.getPitches()[0u] % 16u == 0u &&
         c.getPitches()[0u] % 16u == 0u &&
         reinterpret_cast<std::uintptr_t>(a.data()) % 16u == 0u &&
         reinterpret_cast<std::uintptr_t>(b.data()) % 16u == 0u &&
         reinterpret_cast<std::uintptr_t>(c.data()) % 16u == 0u;
}
} // namespace internal

template <typename Selected, alpaka::onHost::concepts::Device Device,
          alpaka::concepts::QueuePolicyList Policies>
void enqueueCompiled(Selected,
                     alpaka::onHost::Queue<Device, Policies> const &queue,
                     alpaka::concepts::Executor auto executor,
                     alpaka::concepts::IBuffer auto const &a,
                     alpaka::concepts::IBuffer auto const &b,
                     alpaka::concepts::IBuffer auto &c) {
  constexpr auto threads =
      alpaka::exec::isSeqExecutor_v<ALPAKA_TYPEOF(executor)>
          ? 1u
          : Selected::numThreads;
  queue.enqueue(
      alpaka::onHost::ThreadSpec{alpaka::CVec<Index, Selected::numBlocks>{},
                                 alpaka::CVec<Index, threads>{}, executor},
      alpaka::KernelBundle{
          MatmulKernel{}, a, b, c, Selected::shape,
          internal::vectorLoads(a, b, c),
          internal::PhysicalConfiguration<Selected, threads>{}});
}

// Original 64x64x16, 4x4-register, single-buffer baseline. Known shapes use
// its compiled full-grid specialization, exactly as in the timing campaign.
template <alpaka::onHost::concepts::Device Device,
          alpaka::concepts::QueuePolicyList Policies>
inline void
enqueueBaseline(alpaka::onHost::Queue<Device, Policies> const &queue,
                alpaka::concepts::Executor auto executor,
                alpaka::concepts::IBuffer auto const &a,
                alpaka::concepts::IBuffer auto const &b,
                alpaka::concepts::IBuffer auto &c) {
  auto const shape = internal::matrixShape(a, b, c);
  if (internal::selectGpuConfiguration(
          "NVIDIA A30", "Cuda", "alpaka::exec::GpuCuda", shape,
          [&](auto selected) {
            constexpr auto s = decltype(selected)::shape;
            using Baseline =
                Configuration<s.m, s.n, s.k, 0, 256, tiles[0].blocks(s)>;
            enqueueCompiled(Baseline{}, queue, executor, a, b, c);
          }))
    return;
  constexpr auto threads =
      alpaka::exec::isSeqExecutor_v<ALPAKA_TYPEOF(executor)> ? 1u : 256u;
  queue.enqueue(
      alpaka::onHost::ThreadSpec{tiles[0].blocks(shape),
                                 alpaka::CVec<Index, threads>{}, executor},
      alpaka::KernelBundle{
          LegacyMatmulKernel{}, a, b, c, shape, internal::vectorLoads(a, b, c),
          internal::PhysicalConfiguration<internal::BaselineConfiguration,
                                          threads>{}});
}

// Enqueue C = A*B and return whether an exact measured winner was selected.
// The caller owns buffers and queue lifetime; buffers must not alias.
template <alpaka::onHost::concepts::Device Device,
          alpaka::concepts::QueuePolicyList Policies>
inline bool enqueueMatmul(alpaka::onHost::Queue<Device, Policies> const &queue,
                          alpaka::concepts::Executor auto executor,
                          alpaka::concepts::IBuffer auto const &a,
                          alpaka::concepts::IBuffer auto const &b,
                          alpaka::concepts::IBuffer auto &c) {
  auto const shape = internal::matrixShape(a, b, c);
  auto const device = queue.getDevice();
  auto launch = [&](auto selected) {
    enqueueCompiled(selected, queue, executor, a, b, c);
  };
  auto matched = false;
  if constexpr (std::same_as<ALPAKA_TYPEOF(device.getApi()),
                             alpaka::api::Host>) {
    auto context =
        CpuContext{device.getDeviceProperties().multiProcessorCount, 1u};
#if ALPAKA_OMP
    if constexpr (std::same_as<ALPAKA_TYPEOF(executor),
                               alpaka::exec::CpuOmpBlocks>)
      context.ompThreads = (omp_get_dynamic() == 0 && omp_in_parallel() == 0)
                               ? static_cast<Index>(omp_get_max_threads())
                               : 0u;
#endif
    matched = internal::selectCpuConfiguration(
        device.getName(), device.getApi().getName(),
        alpaka::onHost::demangledName(executor), shape, context, launch);
  } else {
    matched = internal::selectGpuConfiguration(
        device.getName(), device.getApi().getName(),
        alpaka::onHost::demangledName(executor), shape, launch);
  }
  if (matched)
    return true;
  enqueueBaseline(queue, executor, a, b, c);
  return false;
}
} // namespace alpakaTune::example::matmul
