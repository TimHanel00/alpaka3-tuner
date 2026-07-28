// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "alpakaTune/core/TunerInfo.hpp"
#include "alpakaTune/core/timing/internal/interface.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/core/UniformCudaHip.hpp>
#include <alpaka/core/config.hpp>

#include <utility>

#if ALPAKA_LANG_CUDA
#include <alpaka/core/ApiCudaRt.hpp>
#include <cuda_runtime_api.h>
#endif

#if ALPAKA_LANG_HIP
#include <alpaka/core/ApiHipRt.hpp>
#include <hip/hip_runtime_api.h>
#endif

namespace alpakaTune::detail::timing::unifiedCudaHip {

#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP

template <alpaka::concepts::Api Api> struct NativeRuntime;

#if ALPAKA_LANG_CUDA
template <> struct NativeRuntime<alpaka::api::Cuda> : alpaka::ApiCudaRt {
  static auto elapsedMilliseconds(float *milliseconds, Event_t start,
                                  Event_t end) {
    return cudaEventElapsedTime(milliseconds, start, end);
  }
};
#endif

#if ALPAKA_LANG_HIP
template <> struct NativeRuntime<alpaka::api::Hip> : alpaka::ApiHipRt {
  static auto elapsedMilliseconds(float *milliseconds, Event_t start,
                                  Event_t end) {
    return hipEventElapsedTime(milliseconds, start, end);
  }
};
#endif

template <alpaka::concepts::Api Api> class NativeEvent {
  using Interface = NativeRuntime<Api>;

public:
  explicit NativeEvent(int deviceIndex) : m_deviceIndex(deviceIndex) {
    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(Interface,
                                     Interface::setDevice(m_deviceIndex));
    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
        Interface,
        Interface::eventCreateWithFlags(&m_event, Interface::eventDefault));
  }

  NativeEvent(NativeEvent const &) = delete;
  auto operator=(NativeEvent const &) -> NativeEvent & = delete;
  NativeEvent(NativeEvent &&other) noexcept
      : m_deviceIndex(other.m_deviceIndex),
        m_event(std::exchange(other.m_event, typename Interface::Event_t{})) {}

  auto operator=(NativeEvent &&other) noexcept -> NativeEvent & {
    if (this != &other) {
      destroy();
      m_deviceIndex = other.m_deviceIndex;
      m_event = std::exchange(other.m_event, typename Interface::Event_t{});
    }
    return *this;
  }

  ~NativeEvent() noexcept { destroy(); }

  [[nodiscard]] auto get() const noexcept { return m_event; }

private:
  void destroy() noexcept {
    if (m_event == typename Interface::Event_t{})
      return;
    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(
        Interface, Interface::setDevice(m_deviceIndex));
    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(Interface,
                                              Interface::eventDestroy(m_event));
    m_event = typename Interface::Event_t{};
  }

  int m_deviceIndex;
  typename Interface::Event_t m_event{};
};

/** CUDA/HIP device-event timer using an Alpaka non-blocking launch queue. */
template <alpaka::concepts::Api Api, alpaka::onHost::concepts::Device Device>
class KernelTimer {
  using Interface = NativeRuntime<Api>;

public:
  explicit KernelTimer(Device &device)
      : m_boundaryEvent(device.makeEvent()),
        m_startEvent(device.getNativeHandle()),
        m_endEvent(device.getNativeHandle()) {}

  [[nodiscard]] static constexpr auto measurementSource() noexcept
      -> RuntimeMeasurementSource {
    return RuntimeMeasurementSource::deviceEvent;
  }

  [[nodiscard]] auto measure(auto const &queue, auto &&launch) -> double {
    static_assert(std::same_as<ALPAKA_TYPEOF(queue.getQueueKind()),
                               alpaka::queueKind::NonBlocking>,
                  "CUDA/HIP device timing requires the non-blocking queue "
                  "returned by alpakaTune::makeQueue with timing::enabled");
    queue.enqueue(m_boundaryEvent);
    alpaka::onHost::wait(m_boundaryEvent);

    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
        Interface,
        Interface::eventRecord(m_startEvent.get(), queue.getNativeHandle()));
    ALPAKA_FORWARD(launch)(queue);
    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
        Interface,
        Interface::eventRecord(m_endEvent.get(), queue.getNativeHandle()));
    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
        Interface, Interface::eventSynchronize(m_endEvent.get()));

    auto milliseconds = 0.0F;
    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
        Interface, Interface::elapsedMilliseconds(
                       &milliseconds, m_startEvent.get(), m_endEvent.get()));
    return static_cast<double>(milliseconds) * 1.0e-3;
  }

private:
  ALPAKA_TYPEOF(std::declval<Device &>().makeEvent()) m_boundaryEvent;
  NativeEvent<Api> m_startEvent;
  NativeEvent<Api> m_endEvent;
};

#endif

} // namespace alpakaTune::detail::timing::unifiedCudaHip

namespace alpakaTune::detail::timing::internal {

#if ALPAKA_LANG_CUDA
template <>
struct MakeKernelTimer::Op<alpaka::api::Cuda, alpaka::deviceKind::NvidiaGpu> {
  [[nodiscard]] auto
  operator()(alpaka::onHost::concepts::Device auto &device) const {
    return unifiedCudaHip::KernelTimer<alpaka::api::Cuda,
                                       ALPAKA_TYPEOF(device)>{device};
  }
};

template <>
struct MakeTuningQueue::Op<alpaka::api::Cuda, alpaka::deviceKind::NvidiaGpu> {
  [[nodiscard]] auto operator()(alpaka::onHost::concepts::Device auto &device,
                                alpaka::concepts::QueueKind auto kind,
                                ::alpakaTune::timing::Enabled) const {
    return device.makeQueue(kind);
  }
};
#endif

#if ALPAKA_LANG_HIP
template <>
struct MakeKernelTimer::Op<alpaka::api::Hip, alpaka::deviceKind::AmdGpu> {
  [[nodiscard]] auto
  operator()(alpaka::onHost::concepts::Device auto &device) const {
    return unifiedCudaHip::KernelTimer<alpaka::api::Hip, ALPAKA_TYPEOF(device)>{
        device};
  }
};

template <>
struct MakeTuningQueue::Op<alpaka::api::Hip, alpaka::deviceKind::AmdGpu> {
  [[nodiscard]] auto operator()(alpaka::onHost::concepts::Device auto &device,
                                alpaka::concepts::QueueKind auto kind,
                                ::alpakaTune::timing::Enabled) const {
    return device.makeQueue(kind);
  }
};
#endif

} // namespace alpakaTune::detail::timing::internal
