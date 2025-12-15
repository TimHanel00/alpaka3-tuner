/* Copyright 2025 Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once
#include "alpakaTune/config/Config.hpp"
#include "alpakaTune/config/ConfigRecord.hpp"
#include "alpakaTune/core/TuningContextManager.hpp"
#include "alpakaTune/store/KernelTuningMetadata.hpp"

#include <cstdint> // uint32_t
#include <ostream> // std::ostream
#include <sstream> // std::stringstream
#include <string>  // std::string

namespace aTune {

struct Info {
  internal::store::KernelTuningMetadata const m_metaData;
  uint64_t const m_numberOfKernelExecutions;
  uint64_t const m_totalNumberOfConfigsEvaluated;
  std::optional<uint64_t> const m_maxNumberOfKernelInvocations;
  uint64_t const m_maxTotalNumberOfConfigs;
  uint64_t const m_numberOfValidConfigsEvaluated;
  uint64_t const m_averageNumberOfExecutionsPerConfig;
  uint64_t const m_averageMeasurementPerConfig;
  uint64_t const m_medianMeasurementPerConfig;
  uint64_t const m_minMeasurementPerConfig;
  uint64_t const m_maxMeasurementPerConfig;
  std::string const bestConfig;
  bool validContext = true;
  std::string printBestConfig() const { return bestConfig; }
  std::string toString() const {
    std::stringstream str;
    str << std::string(80, '-') << std::endl;
    str << "Tuner INFO:" << std::endl;
    str << "    Context:" << std::endl;
    str << m_metaData.printWithIntendation(7) << std::endl;
    if (m_maxNumberOfKernelInvocations.has_value()) {
      str << "    Number of kernel executions: " << m_numberOfKernelExecutions
          << " vs max possible number of executions "
          << m_maxNumberOfKernelInvocations.value() << std::endl;
    } else {
      str << "    Number of kernel executions: " << m_numberOfKernelExecutions
          << std::endl;
    }
    str << "    Total number of configs generated (so far): "
        << m_totalNumberOfConfigsEvaluated
        << " out of possible: " << m_maxTotalNumberOfConfigs << std::endl;
    str << "    Number of valid configs generated "
        << m_numberOfValidConfigsEvaluated << std::endl;
    str << "    Average number of measurements per config "
        << m_averageNumberOfExecutionsPerConfig << std::endl;
    if (!validContext) {
      str << "    WARNING: No measurements taken yet for this context! "
             "Recommended steps:"
          << std::endl;
      str << "                 - increasing the number of enqueues"
          << std::endl;
      str << "                 - decrease the number of warm-up-runs required "
             "-- cmakeFlag: alpakaTune_WARMUP_RUNS"
          << std::endl;
      str << std::string(80, '-') << std::endl;
      return str.str();
    }
    str << "    Measurements per config (in nanoseconds if timing metric "
           "interface): "
        << std::endl;
    str << "     Avg: " << m_averageMeasurementPerConfig
        << " | Median: " << m_medianMeasurementPerConfig
        << " | Min: " << m_minMeasurementPerConfig
        << " | Max: " << m_maxMeasurementPerConfig << std::endl;
    str << "    " << bestConfig << std::endl;
    str << std::string(80, '-') << std::endl;
    return str.str();
  };
  friend std::ostream &operator<<(std::ostream &os, const Info &info) {
    return os << info.toString();
  };
};
namespace internal {
template <typename T_Base>
std::string
getBestConfig(core::TuningContextManager<T_Base> &tuningContextManager) {
  auto record = tuningContextManager.env_environmentState.getBestConfig();
  if (!record)
    return "No best config found";
  auto &model = tuningContextManager.env_tuningModel;
  auto tuple = model.getValuesFromConfig(record.value().get().m_config);
  std::string ret = "Best Config found: \n";
  meta::forEach(tuple, [&]<typename T, auto Id, auto kind>(
                           ParameterAccessor<T, Id, kind> &parameterAccessor) {
    ret += "          " + parameterAccessor.toString();
  });
  return ret;
}
template <typename T_Base>
Info createInfoStruct(
    core::TuningContextManager<T_Base> &tuningContextManager) {
  uint64_t arg_numExec =
      tuningContextManager.env_environmentState.numKernelInvocations;
  uint64_t arg_numConfigs =
      tuningContextManager.env_environmentState.numberOfCheckedConfigs;
  std::optional<uint64_t> arg_maxExec =
      tuningContextManager.env_environmentState.maxNumKernelInvocations;
  uint64_t arg_maxConfigs =
      tuningContextManager.env_environmentState.maxConfigsTotal;
  uint64_t arg_numValid =
      tuningContextManager.env_environmentState.numValidConfigs;
  std::vector<uint64_t> collectedMedians;
  std::vector<uint64_t> numberMeasurements(
      tuningContextManager.env_activeHistory.size());
  uint64_t index{0};
  for (auto entry : tuningContextManager.env_activeHistory.getAll()) {
    auto configRecord = entry.second;
    numberMeasurements[index] = configRecord.getRunCount();
    if (static_cast<uint32_t>(configRecord.state) <=
        static_cast<uint32_t>(config::ConfigState::WarmUp)) {
      index++;
      continue;
    }
    collectedMedians.push_back(static_cast<uint64_t>(configRecord.getMedian()));
    index++;
  }

  uint64_t arg_averageNumberOfExecutionsPerConfig =
      std::accumulate(numberMeasurements.begin(), numberMeasurements.end(),
                      uint64_t{0}) /
      numberMeasurements.size();
  // check any valid config measured
  if (collectedMedians.size() == 0) {
    auto info = Info{tuningContextManager.env_metaData,
                     arg_numExec,
                     arg_numConfigs,
                     arg_maxExec,
                     arg_maxConfigs,
                     arg_numValid,
                     arg_averageNumberOfExecutionsPerConfig};
    info.validContext = false;
    return info;
  }
  std::ranges::sort(collectedMedians);
  uint64_t arg_avgTimePerConfigs =
      std::accumulate(collectedMedians.begin(), collectedMedians.end(),
                      uint64_t{0}) /
      collectedMedians.size();
  uint64_t arg_medianTimePerConfig =
      collectedMedians[collectedMedians.size() / 2 -
                       (collectedMedians.size() % 2 == 0)];

  uint64_t arg_minConfigExecTime = collectedMedians.front();
  uint64_t arg_maxConfigExecTime = collectedMedians.back();
  std::string arg_bestConfig = getBestConfig(tuningContextManager);

  return Info{tuningContextManager.env_metaData,
              arg_numExec,
              arg_numConfigs,
              arg_maxExec,
              arg_maxConfigs,
              arg_numValid,
              arg_averageNumberOfExecutionsPerConfig,
              arg_avgTimePerConfigs,
              arg_medianTimePerConfig,
              arg_minConfigExecTime,
              arg_maxConfigExecTime,
              arg_bestConfig,
              true};
}

} // namespace internal
} // namespace aTune
