// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0
#pragma once

#include "PiKernel.hpp"
#include <array>
#include <chrono>
#include <exception>
#include <functional>
#include <stdexcept>
#include <utility>

namespace adaptivePi {
inline void require(bool condition, char const *message) {
  if (!condition)
    throw std::runtime_error{message};
}
struct Calculation {
  double estimate{}, lowerBound{}, upperBound{}, seconds{};
  double partitionArea{};
  std::size_t insideTiles{}, outsideTiles{}, splitTiles{}, sampledTiles{},
      randomPoints{};
  std::uint32_t deepestSample{};
  std::array<std::uint32_t, 2u> parameters{};
};

template <alpaka::onHost::concepts::Device Device,
          alpaka::concepts::Executor Executor>
class Calculator {
public:
  Calculator(Device device, Executor executor)
      : m_device(std::move(device)), m_executor(executor),
        m_queue(m_device.makeQueue()),
        m_root(alpaka::onHost::allocHost<std::uint8_t>(TileIndex{1u, 1u})),
        m_active(alpaka::onHost::alloc<std::uint8_t>(m_device, tileExtents)),
        m_next(alpaka::onHost::allocLike(m_device, m_active)),
        m_tileContributions(
            alpaka::onHost::alloc<double>(m_device, tileExtents)),
        m_leaves(alpaka::onHost::alloc<Leaf>(m_device, tileExtents)),
        m_reduced(alpaka::onHost::alloc<double>(m_device, TileIndex{1u, 1u})),
        m_area(alpaka::onHost::allocHost<double>(TileIndex{1u, 1u})) {}

  ~Calculator() {
    // Finish asynchronous work before the owning buffers are released.
    alpaka::onHost::wait(m_queue);
  }

  auto calculate(std::uint32_t splits, std::uint32_t points) -> Calculation {
    require(splits <= maximumSplitDepth && points > 0u &&
                points <= maximumSamplesPerTile,
            "Invalid adaptive Pi calculation parameters");
    // Measure one complete calculation: root setup, all parallel phases,
    // device refinement, sampling and reduction. Allocation of reusable
    // buffers belongs to Calculator construction and is outside this budget.
    auto const start = std::chrono::steady_clock::now();
    m_root[TileIndex{0u, 0u}] = 1u;
    alpaka::onHost::memset(m_queue, m_active, std::uint8_t{0u});
    alpaka::onHost::memset(m_queue, m_tileContributions, std::uint8_t{0u});
    alpaka::onHost::memset(m_queue, m_leaves, std::uint8_t{0u});
    alpaka::onHost::memcpy(m_queue, m_active, m_root, TileIndex{1u, 1u});
    auto const frame = alpaka::onHost::FrameSpec{
        TileIndex{16u, 8u}, TileIndex{8u, 16u}, m_executor};

    for (std::uint32_t depth = 0u; depth <= splits; ++depth) {
      std::size_t const width = std::size_t{1u} << depth;
      alpaka::onHost::memset(m_queue, m_next, std::uint8_t{0u});
      if (depth < splits)
        m_queue.enqueue(
            frame, alpaka::KernelBundle{RefineTiles{}, m_active.getMdSpan(),
                                        m_next.getMdSpan(),
                                        m_tileContributions.getMdSpan(),
                                        m_leaves.getMdSpan(), width, depth});
      else
        m_queue.enqueue(frame,
                        alpaka::KernelBundle{
                            SelectBoundaryTiles{}, m_active.getMdSpan(),
                            m_next.getMdSpan(), m_tileContributions.getMdSpan(),
                            m_leaves.getMdSpan(), width, depth});
      std::swap(m_active, m_next);
    }
    m_queue.enqueue(
        frame, alpaka::KernelBundle{SampleTiles{}, m_active.getMdSpan(),
                                    m_tileContributions.getMdSpan(),
                                    m_leaves.getMdSpan(),
                                    std::size_t{1u} << splits, splits, points});
    alpaka::onHost::reduce(m_queue, m_executor, 0.0, m_reduced, std::plus{},
                           m_tileContributions);
    alpaka::onHost::memcpy(m_queue, m_area, m_reduced);
    alpaka::onHost::wait(m_queue);

    auto calculation = Calculation{};
    calculation.estimate = 4.0 * m_area[TileIndex{0u, 0u}];
    calculation.parameters = {splits, points};
    calculation.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    last = calculation;
    return calculation;
  }
  Calculation last{};
  std::exception_ptr failure{};

  // Optional inspection is separate from timed calculations. The example
  // main does not transfer this buffer; integration tests inspect actual
  // leaves.
  auto readLeaves() {
    auto host = alpaka::onHost::allocHost<Leaf>(tileExtents);
    alpaka::onHost::memcpy(m_queue, host, m_leaves);
    alpaka::onHost::wait(m_queue);
    return host;
  }

private:
  inline static constexpr auto tileExtents =
      TileIndex{tilesPerAxis, tilesPerAxis};
  Device m_device;
  Executor m_executor;
  alpaka::onHost::Queue<Device> m_queue;
  ALPAKA_TYPEOF(alpaka::onHost::allocHost<std::uint8_t>(TileIndex{1u, 1u}))
  m_root;
  ALPAKA_TYPEOF(alpaka::onHost::alloc<std::uint8_t>(m_device, tileExtents))
  m_active;
  ALPAKA_TYPEOF(m_active) m_next;
  ALPAKA_TYPEOF(alpaka::onHost::alloc<double>(m_device, tileExtents))
  m_tileContributions;
  ALPAKA_TYPEOF(alpaka::onHost::alloc<Leaf>(m_device, tileExtents)) m_leaves;
  ALPAKA_TYPEOF(alpaka::onHost::alloc<double>(m_device, TileIndex{1u, 1u}))
  m_reduced;
  ALPAKA_TYPEOF(alpaka::onHost::allocHost<double>(TileIndex{1u, 1u})) m_area;
};

// This outer kernel is exclusively a host orchestration entry point. Keeping
// it host-only also allows NVCC to register the inner CUDA tile kernel
// normally.
struct RunCalculation {
  template <typename Engine>
  ALPAKA_FN_HOST void operator()(alpaka::onAcc::concepts::Acc auto const &,
                                 Engine *engine, std::uint32_t splits,
                                 std::uint32_t points) const {
    try {
      engine->calculate(splits, points);
    } catch (...) {
      // Propagate workflow failures after the host queue has completed.
      engine->failure = std::current_exception();
    }
  }
};

} // namespace adaptivePi
