/* Copyright 2025 Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once
#include <alpaka/KernelBundle.hpp>
#include <alpaka/onHost/demangledName.hpp>

#include <alpakaTune/core/peripherals/SessionSpecs.hpp>
#include <string>
#include <vector>

namespace aTune::internal::store {
/**
 * @brief Metadata snapshot for a specific kernel tuning context.
 *
 * Holds immutable(ish) descriptive information
 * (device/executor/kernel/metric/specifiers) and bookkeeping for the explored
 * configuration space.
 *
 */
struct KernelTuningMetadata {
  /// Human-readable identifiers for this context.
  std::string
      device; ///< "The alpaka device as a string (demangled name)", etc.
  std::string executor; ///< "The alpaka executor as a string (demangled name)".
  std::string kernel;   ///< "demangled kernel name".
  std::string targetMetric; ///< primary optimization target, e.g. "time".
  std::string kernelArgs;   ///< argument Types of the kernelBundle (demangled)
  SessionSpecs m_sessionSpecs; ///< Session/context specifiers/tags and
                               ///< potentially break criteria
  std::string printWithIntendation(uint32_t intendation) const {
    std::stringstream os;
    auto concatVec = [](auto const &vec) {
      std::string ret;
      for (auto &elem : vec) {
        ret = ret + elem + ", ";
      };
      if (ret.size() > 2) {
        ret = ret.substr(0, ret.size() - 2);
      }
      return ret;
    };
    std::string intend = std::string(intendation, ' ');
    os << intend << "device: " << this->device << '\n'
       << intend << "executor: " << this->executor << '\n'
       << intend << "kernel: " << this->kernel << '\n'
       << intend << "targetMetric: " << this->targetMetric << '\n'
       << intend << "contextSpecifiers: "
       << concatVec(this->m_sessionSpecs.m_conxtextSpecifiers) << '\n';
    return os.str();
  }
};

/**
 * @brief Build a metadata snapshot from a kernel tuning model.
 *
 * Extracts a parameter accessor from the model (using an empty config) and
 * assembles a @ref KernelTuningMetadata with the given identifiers and tags.
 * Wraps a descriptive information for the active context.
 *
 * @param  device             Human-readable device identifier.
 * @param  exec               Executor/mapping identifier.
 * @param  bundle             Kernel/bundle name.
 * @param  sessionSpecs       Session-level specifiers/tags.
 * @param  targetMetric       Primary optimization target (default: "time").
 * @return KernelTuningMetadata<Config, ParameterAccessor> initialized with
 * descriptors and labels.
 */
template <template <class...> class Bundle, typename T_Kernel,
          typename... T_Args>
auto createTuningMetaData(std::string const &device, std::string const &exec,
                          Bundle<T_Kernel, T_Args...> const &bundle,
                          SessionSpecs const &sessionSpecs,
                          std::string const &targetMetric = "time") {
  std::string argTuple = alpaka::onHost::demangledName<
      typename alpaka::KernelBundle<T_Kernel, T_Args...>::ArgTuple>();

  argTuple.replace(0, std::min<std::size_t>(argTuple.size() - 1, 14), "");
  if (argTuple.size() > 0)
    argTuple.pop_back();
  std::string kernelName = alpaka::onHost::demangledName<T_Kernel>();
  return KernelTuningMetadata{device,       exec,     kernelName,
                              targetMetric, argTuple, sessionSpecs};
};
} // namespace aTune::internal::store
