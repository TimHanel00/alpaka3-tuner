/* Copyright 2025 Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once
#include "alpakaTune/utils/processVariadicArgs.hpp"
#include <cstdint>
#include <optional>
#include <string>
namespace aTune {
/**
 * @brief Session specifications builder.
 *
 * This struct holds optional !runtime! constraints (break criteria) and
 * specifiers for a tuning session:
 * - Maximum runs per configuration
 * - Maximum total execution count
 * - Optional serialized "specifier" string, which allows restricting the tuning
 * context for individual sessions (example: create a context for each buffer
 * size -- alpakaTune will optimize parameters for each buffer size seperately )
 * All setters return `*this` to support a builders-tyle interface.
 * These session specific values can be overwritten using enviroment variables:
 * TunerMaxExecutions and TunerRunsPerConfig
 */
struct SessionSpecs {

  /**
   * Sets the number of *measured* runs per parameter configuration.
   *
   * Warm-up runs are excluded from this count.
   * This value may be overridden if the build is configured with
   * `fastConfigEval` enabled.
   *
   * @param runsPerConfig Maximum number of measured executions per config.
   * @return Reference to `*this` for chaining.
   */
  SessionSpecs &withRunsPerConfig(std::uint32_t runsPerConfig) {
    this->m_runsPerConfig = runsPerConfig;
    return *this;
  }

  /**
   * @brief Set a limit on the total kernel executions.
   *
   * After this many kernel invocations, the tuner enters a
   * "best" mode where no further parameter configurations are evaluated.
   *
   * @param maxExecutions Maximum total executions.
   * @return Reference to `*this` for chaining.
   */
  SessionSpecs &withMaxExecutions(std::uint32_t maxExecutions) {
    this->m_maxExecutions = maxExecutions;
    return *this;
  }
  /**
   * @brief Add session specifiers for contextual tuning.
   *
   * Session specifiers allow creating separate tuning contexts
   * (e.g., per device, problem size, or algorithmic configuration).
   *
   * Each specifier should be convertible to `std::string` (for example via
   * `std::to_string()`).
   *
   * @param specifiers Arbitrary list of specifier values.
   * @return Reference to the current builder.
   */
  template <typename... T_Specifiers>
  SessionSpecs &withContextSpecifier(T_Specifiers... specifiers) {
    internal::processArgs(m_conxtextSpecifiers, specifiers...);
    return *this;
  }

  std::optional<std::uint32_t> m_runsPerConfig;
  std::optional<std::uint32_t> m_maxExecutions;
  std::vector<std::string> m_conxtextSpecifiers;
};
} // namespace aTune
