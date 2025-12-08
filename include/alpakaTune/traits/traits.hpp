/* Copyright 2025 Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once
#include "alpakaTune/tunable/tunables.hpp"

namespace aTune::trait {
struct postProcessing {
  template <typename T_Config, typename T_FrameSpec, typename T_Metric,
            typename KernelFn, typename... args>
  struct Op {
    void operator()(T_Config &config, T_FrameSpec &frame_spec,
                    T_Metric &metricInterface,
                    alpaka::KernelBundle<KernelFn, args...> const &kernel) {}
  };
};

template <typename T_KernelBundle, typename T_Config, typename T_FrameSpec,
          typename T_Metric>
auto callPostProcessing(

    T_Config &config, T_FrameSpec &frame_spec, T_Metric &metricInterface,
    T_KernelBundle &KernelBundle) {
  return postProcessing::Op<T_Config, T_FrameSpec, T_Metric, T_KernelBundle>{}(

      config, frame_spec, metricInterface, KernelBundle);
}

// default
struct preProcessing {
  template <typename T_Config, typename T_FrameSpec, typename T_Metric,
            typename KernelFn, typename... args>
  struct Op {
    void operator()(T_Config &config, T_FrameSpec &frame_spec,
                    T_Metric &metricInterface,
                    alpaka::KernelBundle<KernelFn, args...> const &kernel) {}
  };
};

template <typename T_KernelBundle, typename T_Config, typename T_FrameSpec,
          typename T_Metric>
auto callPreProcessing(

    T_Config &config, T_FrameSpec &frame_spec, T_Metric &metricInterface,
    T_KernelBundle const &KernelBundle) {
  return preProcessing::Op<T_Config, T_FrameSpec, T_Metric, T_KernelBundle>{}(
      config, frame_spec, metricInterface, KernelBundle);
}

template <typename Kernel> struct CompileTimeTuneableTrait {
  // a Kernel can have several template parameters (some of which may not be
  // tuneable)
  // -- tuned_indices is a CVector that holds a list of positions for tuneables
  // defined in tuneAbleDefinitions()
  static constexpr auto tuned_indices =
      alpaka::CVec<std::size_t, static_cast<std::size_t>(0)>{};

  //-> assert the tuned_indicies is of
  static auto tuneAbleDefinitions() {
    return std::tuple{}; // empty tuple, no tunables
  }
};

template <typename T, typename> struct Serialize {
  std::string operator()(T const &) const = delete;
};

// Primary template
template <typename Kernel, typename = void>
struct hasUserDefinedCTuneable : std::false_type {};

// Specialization when tuneAbleDefinitions() is valid
template <typename Kernel>
struct hasUserDefinedCTuneable<
    Kernel, std::void_t<decltype(CompileTimeTuneableTrait<
                                 Kernel>::tuneAbleDefinitions())>> {
private:
  using TupleType =
      decltype(CompileTimeTuneableTrait<Kernel>::tuneAbleDefinitions());

public:
  static constexpr bool value = !meta::is_empty_tuple<TupleType>::value;
};

} // namespace aTune::trait

namespace alpaka::trait {
// a tunable is marked as trivially copyable if the underlying type is copyable
// (types get extracted prior to alpaka::enqueue)
template <auto Tag, typename T>
struct IsKernelArgumentTriviallyCopyable<aTune::Tunable<Tag, T>>
    : std::bool_constant<std::is_trivially_copyable_v<T>> {};

// a tunableMD is marked as trivially copyable if the underlying type is
// copyable (types get extracted prior to alpaka::enqueue)
template <auto Tag, typename Vec>
struct IsKernelArgumentTriviallyCopyable<aTune::TunableMD<Tag, Vec>>
    : std::bool_constant<std::is_trivially_copyable_v<Vec>> {};

// a Ctunable is marked as trivially copyable if the all of its elements are
// copyable (types get extracted prior to alpaka::enqueue)
template <std::uint32_t ID, typename... Ts>
struct IsKernelArgumentTriviallyCopyable<aTune::CTunable<ID, Ts...>>
    : std::bool_constant<(std::is_trivially_copyable_v<Ts> && ...)> {};

} // namespace alpaka::trait
