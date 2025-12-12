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
using integerType = std::uint32_t;

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
  integerType maxExecutions = std::numeric_limits<integerType>::max();

  /// Maximum number of times a single parameter configuration is executed
  /// (warmup excluded — those are not counted).
  integerType runsPerConfig =
      static_cast<integerType>(upperBoundForRunsPerConfig);

  // -------------------------------------------------------------------------
  // Env guards: true → env provided value → setters are locked
  // -------------------------------------------------------------------------

  bool has_MaxExecutions = false;
  bool has_RunsPerConfig = false;

  bool inited = false;

  EnvVarsManager() { initFromEnvOnce(); }

  static EnvVarsManager &get() {
    static EnvVarsManager singleton;
    return singleton;
  }

  /**
   * @brief Parse an unsigned 32-bit integer from environment variable.
   */
  static std::optional<integerType> parseEnvU32(char const *name) {
    if (char const *var = std::getenv(name)) {
      try {
        return static_cast<integerType>(std::stoul(var));
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

    // ------------------- NEW VARIABLE -------------------
    if (auto v = parseEnvU32("TunerMaxExecutions")) {
      maxExecutions = v.value_or(maxExecutions);
      has_MaxExecutions = true;
    }

    // ------------------- EXISTING VARIABLE --------------
    if (auto v = parseEnvU32("TunerRunsPerConfig")) {
      runsPerConfig = v.value_or(runsPerConfig);
      has_RunsPerConfig = true;
    }
  }
};

// ---------------- Flags ----------------
inline bool hasMaxExecutions_Env() {
  return EnvVarsManager::get().has_MaxExecutions;
}

inline bool hasRunsPerConfig_Env() {
  return EnvVarsManager::get().has_RunsPerConfig;
}
/// get `TunerMaxExecutions` -- the maximum number of times the kernel is
/// invoked before running with the best configuration
inline integerType getMaxExecutions() {
  return EnvVarsManager::get().maxExecutions;
}
/// get `TunerRunsPerConfig` -- the maximum number of times the performence of a
/// single parameter configuration is measured (excluding warm-up runs).
inline integerType getRunsPerConfig() {
  return EnvVarsManager::get().runsPerConfig;
}

/// set `TunerMaxExecutions` -- the maximum number of times the kernel is
/// invoked before running with the best configuration
inline bool setMaxExecutions(integerType v) {
  auto &s = EnvVarsManager::get();
  if (s.has_MaxExecutions)
    return false;
  s.maxExecutions = v;
  return true;
}
/// set `TunerRunsPerConfig` -- the maximum number of times the performence of a
/// single parameter configuration is measured (excluding warm-up runs).
inline bool setRunsPerConfig(integerType v) {
  auto &s = EnvVarsManager::get();
  if (s.has_RunsPerConfig)
    return false;
  s.runsPerConfig = v;
  return true;
}
} // namespace aTune::Vars
