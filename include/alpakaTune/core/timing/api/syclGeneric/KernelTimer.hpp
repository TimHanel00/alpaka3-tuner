// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "alpakaTune/core/TunerInfo.hpp"
#include "alpakaTune/core/timing/api/host/KernelTimer.hpp"
#include "alpakaTune/core/timing/internal/interface.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/core/config.hpp>

#if ALPAKA_LANG_ONEAPI
#include <alpaka/api/oneApi/Queue.hpp>
#include <alpaka/api/oneApi/StaticSharedMemory.hpp>
#include <alpaka/core/syclConfig.hpp>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#endif

namespace alpakaTune::detail::timing::syclGeneric {

#if ALPAKA_LANG_ONEAPI

/** Profiling-enabled SYCL event produced by the temporary measurement queue. */
class TimedEvent {
public:
  explicit TimedEvent(sycl::event event) : m_event(std::move(event)) {}

  [[nodiscard]] auto elapsedSeconds() const -> double {
    m_event.wait_and_throw();
    auto const start = m_event.template get_profiling_info<
        sycl::info::event_profiling::command_start>();
    auto const end = m_event.template get_profiling_info<
        sycl::info::event_profiling::command_end>();
    return static_cast<double>(end - start) * 1.0e-9;
  }

private:
  sycl::event m_event;
};

/**
 * Minimal Alpaka-compatible queue for timing SYCL GPU kernels.
 *
 * This compatibility bridge duplicates only Alpaka's OneAPI kernel submission
 * path so alpakaTune can request `enable_profiling`. Remove it when Alpaka's
 * public queue/event interface provides profiling events.
 */
template <alpaka::onHost::concepts::Device Device> class ProfilingQueue {
public:
  explicit ProfilingQueue(Device device)
      : m_device(std::move(device)), m_queue(makeNativeQueue(m_device)) {}

  void enqueue(alpaka::onHost::concepts::ThreadSpec auto const &threadSpec,
               alpaka::concepts::KernelBundle auto const &kernelBundle) const {
    enqueueThread<false>(threadSpec, kernelBundle);
  }

  void enqueue(alpaka::onHost::concepts::FrameSpec auto const &frameSpec,
               alpaka::concepts::KernelBundle auto const &kernelBundle) const {
    auto const threadSpec = alpaka::onHost::internal::adjustThreadSpec(
        *m_device.get(), frameSpec, kernelBundle);
    enqueueThread<true>(threadSpec, kernelBundle);
  }

  [[nodiscard]] auto lastTimedEvent() const -> TimedEvent {
    if (!m_lastEvent)
      throw std::logic_error{"No kernel was submitted to the profiling queue"};
    return TimedEvent{*m_lastEvent};
  }

  void wait() const { m_queue.wait_and_throw(); }

  [[nodiscard]] static constexpr auto getQueueKind() noexcept {
    return alpaka::queueKind::nonBlocking;
  }

private:
  [[nodiscard]] static auto makeNativeQueue(Device const &device)
      -> sycl::queue {
    auto const [nativeDevice, nativeContext] = device.getNativeHandle();
    return sycl::queue{
        nativeContext, nativeDevice,
        {sycl::property::queue::in_order{},
         sycl::property::queue::enable_profiling{}}};
  }

  template <bool LaunchedWithFrameSpec>
  void enqueueThread(
      alpaka::onHost::concepts::ThreadSpec auto const &threadSpec,
      alpaka::concepts::KernelBundle auto const &kernelBundle) const {
    constexpr auto staticSharedBytes =
        alpaka::onAcc::oneApi::StaticSharedMemory::sizeLookupBufferInBytes(
            ALPAKA_SYCL_NUM_MAX_SHARED_MEMORY_ALLOCATIONS);
    auto const dynamicSharedBytes = std::max(
        uint32_t{1u},
        alpaka::onHost::getDynSharedMemBytes(threadSpec, kernelBundle));
    auto const [nativeDevice, nativeContext] = m_device.getNativeHandle();
    static_cast<void>(nativeContext);
    if (staticSharedBytes + dynamicSharedBytes >
        nativeDevice
            .template get_info<sycl::info::device::local_mem_size>())
      throw std::runtime_error{
          "Requested shared memory exceeds the SYCL device limit"};

    auto const warpSize = m_device.getDeviceProperties().warpSize;
    auto const api = m_device.getApi();
    auto const deviceKind = m_device.getDeviceKind();
    m_lastEvent = alpaka::onHost::syclGeneric::Warpsize::Dispatch<
        ALPAKA_TYPEOF(deviceKind)>{}(
        deviceKind,
        [&](auto compileTimeWarpSize)
          requires std::same_as<
              std::integral_constant<
                  typename ALPAKA_TYPEOF(compileTimeWarpSize)::value_type,
                  ALPAKA_TYPEOF(compileTimeWarpSize)::value>,
              ALPAKA_TYPEOF(compileTimeWarpSize)>
        {
          return m_queue.submit(
              [api, compileTimeWarpSize, deviceKind, threadSpec, kernelBundle,
               dynamicSharedBytes](sycl::handler &handler) {
                auto staticSharedAccessor = sycl::local_accessor<std::byte>{
                    sycl::range<1>{staticSharedBytes}, handler};
                auto dynamicSharedAccessor = sycl::local_accessor<std::byte>{
                    sycl::range<1>{dynamicSharedBytes}, handler};
                auto const worker =
                    alpaka::onHost::internal::detail::getWorkerDescription(
                        threadSpec);
                constexpr auto syclDim = worker.first.dimensions;
                constexpr auto warp =
                    ALPAKA_TYPEOF(compileTimeWarpSize)::value;
                alpaka::onHost::internal::detail::EnqueueKernelWithWarpSize<
                    syclDim, warp,
                    ALPAKA_SYCL_SUBGROUP_SIZE & warp>::call(
                    handler, worker.first, kernelBundle, staticSharedAccessor,
                    dynamicSharedAccessor, worker.second,
                    alpaka::DictEntry(alpaka::object::api, api),
                    alpaka::DictEntry(alpaka::object::deviceKind,
                                      deviceKind),
                    alpaka::DictEntry(
                        alpaka::object::exec,
                        ALPAKA_TYPEOF(threadSpec)::getExecutor()),
                    alpaka::DictEntry(alpaka::object::launchedWidthFrameSpec,
                                      std::bool_constant<
                                          LaunchedWithFrameSpec>{}),
                    alpaka::DictEntry(alpaka::object::warpSize,
                                      compileTimeWarpSize));
              });
        },
        warpSize);
  }

  Device m_device;
  mutable sycl::queue m_queue;
  mutable std::optional<sycl::event> m_lastEvent;
};

/** SYCL GPU timer backed by command profiling on a dedicated queue. */
template <alpaka::onHost::concepts::Device Device> class KernelTimer {
public:
  explicit KernelTimer(Device &) {}

  [[nodiscard]] static constexpr auto measurementSource() noexcept
      -> RuntimeMeasurementSource {
    return RuntimeMeasurementSource::deviceEvent;
  }

  [[nodiscard]] auto measure(auto const &queue, auto &&launch) -> double {
    queue.wait();
    ALPAKA_FORWARD(launch)(queue);
    return queue.lastTimedEvent().elapsedSeconds();
  }
};

#endif

} // namespace alpakaTune::detail::timing::syclGeneric

namespace alpakaTune::detail::timing::internal {

template <>
struct MakeKernelTimer::Op<alpaka::api::OneApi, alpaka::deviceKind::Cpu> {
  [[nodiscard]] auto operator()(alpaka::onHost::concepts::Device auto &device)
      const {
    return host::KernelTimer<ALPAKA_TYPEOF(device)>{device};
  }
};

template <>
struct MakeTuningQueue::Op<alpaka::api::OneApi,
                           alpaka::deviceKind::Cpu> {
  [[nodiscard]] auto operator()(alpaka::onHost::concepts::Device auto &device,
                                alpaka::concepts::QueueKind auto kind,
                                ::alpakaTune::timing::Enabled) const {
    return device.makeQueue(kind);
  }
};

#if ALPAKA_LANG_ONEAPI
template <alpaka::concepts::GpuType DeviceKind>
struct MakeKernelTimer::Op<alpaka::api::OneApi, DeviceKind> {
  [[nodiscard]] auto operator()(alpaka::onHost::concepts::Device auto &device)
      const {
    return syclGeneric::KernelTimer<ALPAKA_TYPEOF(device)>{device};
  }
};

template <alpaka::concepts::GpuType DeviceKind>
struct MakeTuningQueue::Op<alpaka::api::OneApi, DeviceKind> {
  [[nodiscard]] auto operator()(alpaka::onHost::concepts::Device auto &device,
                                alpaka::queueKind::NonBlocking,
                                ::alpakaTune::timing::Enabled) const {
    return syclGeneric::ProfilingQueue<ALPAKA_TYPEOF(device)>{device};
  }
};
#endif

} // namespace alpakaTune::detail::timing::internal
