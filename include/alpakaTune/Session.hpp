// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "alpakaTune/Strategy.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace alpakaTune {

class ContextBuilder;

struct SessionDefaults {
  std::size_t warmupRuns{1u};
  /** Maximum number of recorded (non-warm-up) runs per configuration. */
  std::size_t runsPerCandidate{1u};
  /** CI convergence may retire a configuration once this many runs exist. */
  std::size_t minimumRunsPerCandidate{1u};
  std::size_t ciCheckInterval{10u};
  double ciZScore{2.576};
  double ciRelativeWidth{0.05};
  double outlierMadScale{3.5};
  bool mannWhitneyEarlyStop{true};
  std::size_t mannWhitneyMinimumSamples{8u};
  double mannWhitneyAlpha{0.05};
  std::size_t noiseCancellationWindow{50u};
  std::size_t maxConsecutiveRuns{3u};
  StrategyKind strategy{StrategyKind::exhaustive};
  std::uint64_t randomSeed{0u};
  std::filesystem::path persistenceDirectory{".alpakaTune"};
};

namespace detail {

inline auto requirePositive(YAML::Node const &node, char const *key)
    -> std::size_t {
  if (!node[key])
    throw std::runtime_error{std::string{"Missing YAML key: "} + key};
  auto const value = node[key].as<std::size_t>();
  if (value == 0u)
    throw std::runtime_error{std::string{"YAML key must be greater than zero: "} + key};
  return value;
}

inline auto optionalPositive(YAML::Node const &node, char const *key,
                             std::size_t fallback) -> std::size_t {
  return node[key] ? requirePositive(node, key) : fallback;
}

inline auto optionalPositiveFinite(YAML::Node const &node, char const *key,
                                   double fallback) -> double {
  if (!node[key])
    return fallback;
  auto const value = node[key].as<double>();
  if (!std::isfinite(value) || value <= 0.0)
    throw std::runtime_error{std::string{"YAML key must be positive and finite: "} + key};
  return value;
}

inline auto optionalProbability(YAML::Node const &node, char const *key,
                                double fallback) -> double {
  if (!node[key])
    return fallback;
  auto const value = node[key].as<double>();
  if (!std::isfinite(value) || value <= 0.0 || value >= 1.0)
    throw std::runtime_error{std::string{"YAML key must be finite and in (0, 1): "} + key};
  return value;
}

inline void rejectUnknown(YAML::Node const &node,
                          std::initializer_list<std::string_view> allowed,
                          std::string_view section) {
  for (auto const &entry : node) {
    auto const key = entry.first.as<std::string>();
    auto const found = std::find(allowed.begin(), allowed.end(), key);
    if (found == allowed.end())
      throw std::runtime_error{"Unknown YAML key in " + std::string{section} + ": " + key};
  }
}

inline auto loadDefaults(std::filesystem::path const &path) -> SessionDefaults {
  YAML::Node root;
  try {
    root = YAML::LoadFile(path.string());
  } catch (YAML::Exception const &error) {
    throw std::runtime_error{"Unable to load alpakaTune YAML configuration '" +
                             path.string() + "': " + error.what()};
  }
  if (!root.IsMap())
    throw std::runtime_error{"alpakaTune YAML configuration must contain a map."};
  rejectUnknown(root, {"schema_version", "tuning", "persistence"}, "root");
  if (!root["schema_version"] || root["schema_version"].as<int>() != 1)
    throw std::runtime_error{"Unsupported alpakaTune YAML schema_version; expected 1."};
  if (!root["tuning"] || !root["tuning"].IsMap())
    throw std::runtime_error{"Missing YAML map: tuning"};
  if (!root["persistence"] || !root["persistence"].IsMap())
    throw std::runtime_error{"Missing YAML map: persistence"};

  auto const tuning = root["tuning"];
  auto const persistence = root["persistence"];
  rejectUnknown(tuning,
                {"strategy", "random_seed", "warmup_runs", "runs_per_candidate",
                 "minimum_runs_per_candidate", "ci_check_interval", "ci_z_score",
                 "ci_relative_width", "outlier_mad_scale", "mann_whitney_early_stop",
                 "mann_whitney_min_samples", "mann_whitney_alpha",
                 "noise_cancellation_window", "max_consecutive_runs"},
                "tuning");
  rejectUnknown(persistence, {"directory"}, "persistence");

  SessionDefaults defaults;
  auto const strategy = tuning["strategy"] ? tuning["strategy"].as<std::string>() : "exhaustive";
  defaults.strategy = strategyFromName(strategy);
  defaults.randomSeed = tuning["random_seed"] ? tuning["random_seed"].as<std::uint64_t>() : 0u;
  defaults.warmupRuns = tuning["warmup_runs"] ? tuning["warmup_runs"].as<std::size_t>() : 1u;
  defaults.runsPerCandidate = requirePositive(tuning, "runs_per_candidate");
  defaults.minimumRunsPerCandidate = optionalPositive(
      tuning, "minimum_runs_per_candidate", defaults.runsPerCandidate);
  if (defaults.minimumRunsPerCandidate > defaults.runsPerCandidate)
    throw std::runtime_error{
        "YAML minimum_runs_per_candidate must not exceed runs_per_candidate."};
  defaults.ciCheckInterval = optionalPositive(tuning, "ci_check_interval", 10u);
  defaults.ciZScore = optionalPositiveFinite(tuning, "ci_z_score", 2.576);
  defaults.ciRelativeWidth =
      optionalPositiveFinite(tuning, "ci_relative_width", 0.05);
  defaults.outlierMadScale = optionalPositiveFinite(tuning, "outlier_mad_scale", 3.5);
  defaults.mannWhitneyEarlyStop =
      tuning["mann_whitney_early_stop"]
          ? tuning["mann_whitney_early_stop"].as<bool>()
          : true;
  defaults.mannWhitneyMinimumSamples =
      optionalPositive(tuning, "mann_whitney_min_samples", 8u);
  defaults.mannWhitneyAlpha =
      optionalProbability(tuning, "mann_whitney_alpha", 0.05);
  defaults.noiseCancellationWindow = requirePositive(tuning, "noise_cancellation_window");
  defaults.maxConsecutiveRuns = requirePositive(tuning, "max_consecutive_runs");
  if (defaults.maxConsecutiveRuns <= defaults.warmupRuns)
    throw std::runtime_error{
        "YAML max_consecutive_runs must exceed warmup_runs so every activation is measured."};
  if (!persistence["directory"] || persistence["directory"].as<std::string>().empty())
    throw std::runtime_error{"YAML persistence.directory must not be empty."};
  defaults.persistenceDirectory = persistence["directory"].as<std::string>();
  return defaults;
}

inline auto canonicalConfigPath(std::filesystem::path path) -> std::filesystem::path {
  if (path.empty())
    throw std::runtime_error{"The alpakaTune YAML configuration path must not be empty."};
  if (!std::filesystem::exists(path))
    throw std::runtime_error{"alpakaTune YAML configuration does not exist: " + path.string()};
  return std::filesystem::weakly_canonical(std::move(path));
}

inline auto configurationCache(std::filesystem::path const &path)
    -> std::shared_ptr<SessionDefaults const> {
  static std::mutex mutex;
  static std::unordered_map<std::string, std::weak_ptr<SessionDefaults const>> cache;
  auto const canonical = canonicalConfigPath(path);
  std::lock_guard lock{mutex};
  auto &cached = cache[canonical.string()];
  if (auto shared = cached.lock())
    return shared;
  auto loaded = std::make_shared<SessionDefaults const>(loadDefaults(canonical));
  cached = loaded;
  return loaded;
}

inline auto defaultConfigurationPath() -> std::filesystem::path {
  if (auto const *configured = std::getenv("ALPAKA_TUNE_CONFIG"); configured != nullptr && configured[0] != '\0')
    return configured;
#ifdef ALPAKA_TUNE_DEFAULT_CONFIG
  return ALPAKA_TUNE_DEFAULT_CONFIG;
#else
  throw std::runtime_error{"No alpakaTune YAML configuration was found. Set ALPAKA_TUNE_CONFIG."};
#endif
}

} // namespace detail

class Session {
public:
  Session() : Session{detail::defaultConfigurationPath()} {}
  explicit Session(std::filesystem::path configurationPath)
      : m_configurationPath(detail::canonicalConfigPath(std::move(configurationPath))),
        m_defaults(detail::configurationCache(m_configurationPath)) {}

  [[nodiscard]] static auto loadFromYaml(std::filesystem::path path) -> Session {
    return Session{std::move(path)};
  }

  [[nodiscard]] auto defaults() const noexcept -> SessionDefaults const & {
    return *m_defaults;
  }

  [[nodiscard]] auto configurationPath() const noexcept -> std::filesystem::path const & {
    return m_configurationPath;
  }

  [[nodiscard]] auto contextBuilder() const -> ContextBuilder;

private:
  std::filesystem::path m_configurationPath;
  std::shared_ptr<SessionDefaults const> m_defaults;
};

[[nodiscard]] inline auto session() -> Session {
  static Session const defaultSession{};
  return defaultSession;
}
[[nodiscard]] inline auto session(std::filesystem::path path) -> Session {
  return Session{std::move(path)};
}

} // namespace alpakaTune
