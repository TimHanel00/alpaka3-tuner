/* Copyright 2025 Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once
#include <cstdint>
#include <cstdlib> // std::getenv
#include <iostream>
#include <limits>
#include <optional>
#include <string> // std::stoul

#ifndef upperBoundForRunsPerConfig
#define upperBoundForRunsPerConfig 100
#endif

namespace aTune::Vars {
using T_IntegerType = std::uint64_t;

/**
 * @namespace aTune::Vars
 * @brief Holds runtime-accessible limits/break criteria
 * Supported environment variables:
 * - `TunerMaxExecutions` -- the maximum number of times the kernel is invoked
 * before running with the best configuration
 * - `TunerRunsPerConfig` -- the maximum number of times the performence of a
 * single parameter configuration is measured (excluding warm-up runs). If an
 * env var is present, it overrides existing SessionSpecifiers
 */
struct EnvVarsManager {

  // -------------------------------------------------------------------------
  // Effective values
  // -------------------------------------------------------------------------

  /// Maximum total executions before switching to “best” mode.
  T_IntegerType maxExecutions = std::numeric_limits<T_IntegerType>::max();

  /// Maximum number of times a single parameter configuration is executed
  /// (warmup excluded — those are not counted).
  T_IntegerType maxRunsPerConfig =
      static_cast<T_IntegerType>(upperBoundForRunsPerConfig);
  /// Minimum number of times a single parameter configuration is executed
  /// (warmup excluded — those are not counted).
  T_IntegerType minRunsPerConfig = static_cast<T_IntegerType>(1);

  // -------------------------------------------------------------------------
  // Env guards: true → env provided value → setters are locked
  // -------------------------------------------------------------------------

  bool hasMaxExecutions = false;
  bool hasMaxRunsPerConfig = false;
  bool hasMinRunsPerConfig = false;

  bool inited = false;

  EnvVarsManager() { initFromEnvOnce(); }

  static EnvVarsManager &get() {
    static EnvVarsManager singleton;
    return singleton;
  }

  /**
   * @brief Parse an unsigned integer from environment variable.
   */
  static std::optional<T_IntegerType> parseEnvU(char const *name) {
    if (char const *var = std::getenv(name)) {
      try {
        return static_cast<T_IntegerType>(std::stoul(var));
      } catch (std::exception const &e) {
        std::cerr << "Invalid value for " << name << ": " << e.what() << '\n';
      }
    }
    return std::nullopt;
  }

  /// @brief Initialize env vars exactly once.
  void initFromEnvOnce() {
    if (inited)
      return;
    inited = true;

    if (auto v = parseEnvU("ATuneMaxExecutions")) {
      maxExecutions = v.value_or(maxExecutions);
      hasMaxExecutions = true;
    }

    if (auto v = parseEnvU("ATuneMaxRunsPerConfig")) {
      maxRunsPerConfig = v.value_or(maxRunsPerConfig);
      hasMaxRunsPerConfig = true;
    }
    if (auto v = parseEnvU("ATuneMinRunsPerConfig")) {
      minRunsPerConfig = v.value_or(minRunsPerConfig);
      hasMinRunsPerConfig = true;
    }
  }
};

// ---------------- Flags ----------------
inline bool hasMaxExecutions_Env() {
  return EnvVarsManager::get().hasMaxExecutions;
}
inline bool hasMinRunsPerConfig_Env() {
  return EnvVarsManager::get().hasMinRunsPerConfig;
}
inline bool hasMaxRunsPerConfig_Env() {
  return EnvVarsManager::get().hasMaxRunsPerConfig;
}
/// get `TunerMaxExecutions` -- the maximum number of times the kernel is
/// invoked before running with the best configuration
inline T_IntegerType getMaxExecutions() {
  return EnvVarsManager::get().maxExecutions;
}
/// get `TunerMinRunsPerConfig` -- the maximum number of times the performence
/// of a single parameter configuration is measured (excluding warm-up runs).
inline T_IntegerType getMinRunsPerConfig() {
  return EnvVarsManager::get().minRunsPerConfig;
}
/// get `TunerMaxRunsPerConfig` -- the maximum number of times the performence
/// of a single parameter configuration is measured (excluding warm-up runs).
inline T_IntegerType getMaxRunsPerConfig() {
  return EnvVarsManager::get().maxRunsPerConfig;
}
} // namespace aTune::Vars
