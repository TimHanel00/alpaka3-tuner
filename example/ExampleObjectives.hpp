// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0
#pragma once

#include <alpakaTune/alpakaTune.hpp>

#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

/** @file
 * @brief Example command line selecting TunerConfig::metric.
 *
 * This header is not part of the alpakaTune library interface. It only maps
 * the examples' --objective options onto the tuner configuration; the tuner
 * itself measures and scores the selected metric.
 */
namespace alpakaTune::example {

/** The process tuner configuration including the command-line metric. */
inline auto tunerConfig() -> TunerConfig & {
  static TunerConfig config = alpakaTune::tunerConfig();
  return config;
}

/** Consume --objective, --counter, --time-weight, --counter-weight,
 * --time-scale-seconds and --counter-scale; leave other arguments in place. */
inline bool consumeObjectiveOptions(int &argc, char **argv,
                                    std::string_view defaultCounter) {
  auto &config = tunerConfig();
  config.weightedMetric.counter = metricFromName(defaultCounter);
  int destination = 1;
  try {
    for (int i = 1; i < argc; ++i) {
      auto const argument = std::string_view{argv[i]};
      auto const separator = argument.find('=');
      auto const name = argument.substr(0u, separator);
      if (name != "--objective" && name != "--counter" &&
          name != "--time-weight" && name != "--counter-weight" &&
          name != "--time-scale-seconds" && name != "--counter-scale") {
        argv[destination++] = argv[i];
        continue;
      }
      auto value = std::string{};
      if (separator != std::string_view::npos)
        value = argument.substr(separator + 1u);
      else if (++i < argc)
        value = argv[i];
      else
        throw std::invalid_argument{"Missing value for " + std::string{name}};
      auto number = [&] {
        std::size_t used = 0u;
        auto const parsed = std::stod(value, &used);
        if (used != value.size())
          throw std::invalid_argument{"Invalid numeric option: " + value};
        return parsed;
      };
      auto &weighted = config.weightedMetric;
      if (name == "--objective")
        config.metric = metricFromName(value);
      else if (name == "--counter")
        weighted.counter = metricFromName(value);
      else if (name == "--time-weight")
        weighted.timeWeight = number();
      else if (name == "--counter-weight")
        weighted.counterWeight = number();
      else if (name == "--time-scale-seconds")
        weighted.timeScale = number();
      else
        weighted.counterScale = number();
    }
    config.validate();
    argc = destination;
    argv[destination] = nullptr;
    return true;
  } catch (std::exception const &error) {
    std::cerr << "Objective options: " << error.what() << '\n';
    return false;
  }
}

inline void printObjectiveHelp() {
  std::cerr
      << "  --objective runtime|elapsed-time|instructions|l2-misses|weighted\n"
      << "  --counter instructions|l2-misses (weighted objective)\n"
      << "  --time-weight 0.5 --counter-weight 0.5\n"
      << "  --time-scale-seconds VALUE --counter-scale VALUE (required for "
         "weighted)\n";
}
} // namespace alpakaTune::example
