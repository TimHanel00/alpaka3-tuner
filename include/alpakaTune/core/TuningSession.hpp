/* Copyright 2025 Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include "alpakaTune/core/SessionBuilder.hpp"
#include "alpakaTune/core/TuningContext.hpp"
#include "alpakaTune/tunable/FrameSpecTuningModel.hpp"

#include <utility>

namespace aTune {
namespace internal::core {
template <typename T_Queue, typename T_Exec, typename T_KernelBundle,
          typename T_FrameSpecTuningModel, typename T_Session>
auto *setupEnqueue(T_Queue const &queue, T_Exec const &exec,
                   T_FrameSpecTuningModel const &frameSpecTune,
                   T_KernelBundle const &kernelBundle,
                   T_Session const &session) {
  static auto *kernelptr =
      getTuningEnvironment(queue, exec, frameSpecTune, kernelBundle, session)
          .get();
  store::KernelTuningMetadata const &data = kernelptr->env_metaData;
  if (session.m_sessionSpecs.m_conxtextSpecifiers !=
      data.m_sessionSpecs.m_conxtextSpecifiers) {
    kernelptr =
        getTuningEnvironment(queue, exec, frameSpecTune, kernelBundle, session)
            .get();
  }
  return kernelptr;
}
} // namespace internal::core

#define maxConsecutiveStrategyRuns 4000

// Check if a m_strategy should be applied and Config should be skipped

// Validate constraint or mark as Invalid

#define WarmUpRuns 1

template <typename T_Strategy = aTune::strategy::RandomSearch,
          concepts::MetricInterface T_MetricInterface =
              aTune::metricInterface::Timing,
          typename T_Constraints = std::tuple<>>
struct TuningSession {
  using MetricType = T_MetricInterface;
  T_Strategy m_strategy;
  T_MetricInterface m_metricInterface;
  T_Constraints m_constraintTuple;
  std::string m_outputFile;
  SessionSpecs m_sessionSpecs;
  TuningSession() = default;

  explicit TuningSession(T_Strategy const &strategy,
                         T_MetricInterface const &interface,
                         T_Constraints const &constraints, std::string config,
                         SessionSpecs const &sessionSpecifiers)
      : m_strategy(strategy), m_metricInterface(interface),
        m_constraintTuple(constraints), m_outputFile(std::move(config)),
        m_sessionSpecs(sessionSpecifiers) {}

  /** @brief Enqueue and Execute a kernel, while performing exactly one step of
   * the tuning process. Mirrors the existing alpaka::enqueue both in terms of
   * structure and utility.
   * @note this will explicitly syncronize the queue (calls queue.wait()
   * internally) after the usual alpaka::enqueue.
   * @param queue
   * @param exec
   * @param frameSpecTune
   * @param kernelBundle the compute kernel and there arguments
   */
  template <typename T_Queue, typename T_FrameSpec, typename T_FramesTune,
            typename T_FrameExtentTune, typename T_ThreadTune,
            typename T_BlockTune>
  auto
  enqueue(T_Queue const &queue, alpaka::concepts::Executor auto const &exec,
          FrameSpecTuningModel<T_FrameSpec, T_FramesTune, T_FrameExtentTune,
                               T_ThreadTune, T_BlockTune> const &frameSpecTune,
          alpaka::concepts::KernelBundle auto const &kernelBundle) {
    return enqueueImpl(queue, exec,
                       std::forward<decltype(frameSpecTune)>(frameSpecTune),
                       kernelBundle);
  }

  /** @brief Enqueue and Execute a kernel, while performing exactly one step of
   * the tuning process. Mirrors the existing alpaka::enqueue both in terms of
   * structure and utility.
   * @note this will explicitly syncronize the queue (calls queue.wait()
   * internally) after the usual alpaka::enqueue.
   * @param queue
   * @param exec
   * @param frameSpec
   * @param kernelBundle the compute kernel and there arguments
   */
  template <typename T_Queue, typename T_NumFrames, typename T_FrameExtent,
            typename T_ThreadSpec>
  auto enqueue(T_Queue const &queue,
               alpaka::concepts::Executor auto const &exec,
               alpaka::onHost::FrameSpec<T_NumFrames, T_FrameExtent,
                                         T_ThreadSpec> const &frameSpec,
               alpaka::concepts::KernelBundle auto const &kernelBundle) {
    return enqueueImpl(
        queue, exec,
        FrameSpecTuningModel{std::forward<decltype(frameSpec)>(frameSpec)},
        kernelBundle);
  }

  //@TODO implement a getBestValues method, implement a function based version
  //(no queue, no exec, no
  // frameSpec,...).
  ~TuningSession() {}

private:
  template <typename T_Queue, typename T_FrameSpec>
  auto enqueueImpl(T_Queue &queue, alpaka::concepts::Executor auto const &exec,
                   T_FrameSpec &&spec,
                   alpaka::concepts::KernelBundle auto const &kernelBundle) {
    auto *environmentPtr = aTune::internal::core::setupEnqueue(
        queue, exec, std::forward<T_FrameSpec>(spec), kernelBundle, *this);

    environmentPtr->launch(queue, exec, std::forward<T_FrameSpec>(spec),
                           kernelBundle);
  }
};

} // namespace aTune
