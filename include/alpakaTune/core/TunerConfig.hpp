// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "alpakaTune/interfaces/Strategy.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace alpakaTune {

struct TunerConfig {
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
  std::optional<std::size_t> maximumExecutions;
  std::optional<std::size_t> maximumRetiredConfigurations;
  StrategyKind strategy{StrategyKind::exhaustive};
  std::uint64_t randomSeed{0u};
  std::filesystem::path persistenceFile{".alpakaTune/history.json"};
  /** Optional override for the model used by the learned-hybrid strategy. */
  std::filesystem::path learnedModelFile{
#ifdef ALPAKA_TUNE_DEFAULT_MODEL
      ALPAKA_TUNE_DEFAULT_MODEL
#endif
  };
  /** Non-learned strategy used when a model cannot be loaded safely. */
  StrategyKind learnedFallback{StrategyKind::random};
  /** Maximum number of unmeasured learned candidates held at once. */
  std::size_t learnedCandidatePoolSize{4'096u};
  /** Maximum number of candidates sent through one learned scoring batch. */
  std::size_t learnedCandidateBatchSize{256u};

  /** Load a mutable tuner configuration from a schema-version-1 or -2 YAML
   * file. */
  [[nodiscard]] static auto fromYaml(std::filesystem::path path) -> TunerConfig;

  /** Reject values which cannot form a valid tuner. */
  void validate() const;
};

namespace detail {

inline auto requirePositive(YAML::Node const &node, char const *key)
    -> std::size_t {
  if (!node[key])
    throw std::runtime_error{std::string{"Missing YAML key: "} + key};
  auto const value = node[key].as<std::size_t>();
  if (value == 0u)
    throw std::runtime_error{
        std::string{"YAML key must be greater than zero: "} + key};
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
    throw std::runtime_error{
        std::string{"YAML key must be positive and finite: "} + key};
  return value;
}

inline auto optionalProbability(YAML::Node const &node, char const *key,
                                double fallback) -> double {
  if (!node[key])
    return fallback;
  auto const value = node[key].as<double>();
  if (!std::isfinite(value) || value <= 0.0 || value >= 1.0)
    throw std::runtime_error{
        std::string{"YAML key must be finite and in (0, 1): "} + key};
  return value;
}

inline void rejectUnknown(YAML::Node const &node,
                          std::initializer_list<std::string_view> allowed,
                          std::string_view section) {
  for (auto const &entry : node) {
    auto const key = entry.first.as<std::string>();
    auto const found = std::find(allowed.begin(), allowed.end(), key);
    if (found == allowed.end())
      throw std::runtime_error{"Unknown YAML key in " + std::string{section} +
                               ": " + key};
  }
}

inline auto loadTunerConfig(std::filesystem::path const &path) -> TunerConfig {
  YAML::Node root;
  try {
    root = YAML::LoadFile(path.string());
  } catch (YAML::Exception const &error) {
    throw std::runtime_error{"Unable to load alpakaTune YAML configuration '" +
                             path.string() + "': " + error.what()};
  }
  if (!root.IsMap())
    throw std::runtime_error{
        "alpakaTune YAML configuration must contain a map."};
  rejectUnknown(root, {"schema_version", "tuning", "persistence", "learning"},
                "root");
  if (!root["schema_version"])
    throw std::runtime_error{"Missing YAML key: schema_version"};
  auto const schemaVersion = root["schema_version"].as<int>();
  if (schemaVersion != 1 && schemaVersion != 2)
    throw std::runtime_error{
        "Unsupported alpakaTune YAML schema_version; expected 1 or 2."};
  if (!root["tuning"] || !root["tuning"].IsMap())
    throw std::runtime_error{"Missing YAML map: tuning"};
  if (!root["persistence"] || !root["persistence"].IsMap())
    throw std::runtime_error{"Missing YAML map: persistence"};

  auto const tuning = root["tuning"];
  auto const persistence = root["persistence"];
  auto const learning = root["learning"];
  if (learning && schemaVersion < 2)
    throw std::runtime_error{
        "YAML learning configuration requires schema_version: 2."};
  if (learning && !learning.IsMap())
    throw std::runtime_error{"YAML learning must contain a map."};
  rejectUnknown(tuning,
                {"strategy", "random_seed", "warmup_runs", "runs_per_candidate",
                 "minimum_runs_per_candidate", "ci_check_interval",
                 "ci_z_score", "ci_relative_width", "outlier_mad_scale",
                 "mann_whitney_early_stop", "mann_whitney_min_samples",
                 "mann_whitney_alpha", "noise_cancellation_window",
                 "max_consecutive_runs", "maximum_executions",
                 "maximum_retired_configurations"},
                "tuning");
  rejectUnknown(persistence, {"file", "directory"}, "persistence");
  if (learning)
    rejectUnknown(learning,
                  {"model", "fallback", "candidate_pool_size",
                   "candidate_batch_size"},
                  "learning");

  TunerConfig defaults;
  auto const strategy =
      tuning["strategy"] ? tuning["strategy"].as<std::string>() : "exhaustive";
  defaults.strategy = strategyFromName(strategy);
  defaults.randomSeed =
      tuning["random_seed"] ? tuning["random_seed"].as<std::uint64_t>() : 0u;
  defaults.warmupRuns =
      tuning["warmup_runs"] ? tuning["warmup_runs"].as<std::size_t>() : 1u;
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
  defaults.outlierMadScale =
      optionalPositiveFinite(tuning, "outlier_mad_scale", 3.5);
  defaults.mannWhitneyEarlyStop =
      tuning["mann_whitney_early_stop"]
          ? tuning["mann_whitney_early_stop"].as<bool>()
          : true;
  defaults.mannWhitneyMinimumSamples =
      optionalPositive(tuning, "mann_whitney_min_samples", 8u);
  defaults.mannWhitneyAlpha =
      optionalProbability(tuning, "mann_whitney_alpha", 0.05);
  defaults.noiseCancellationWindow =
      requirePositive(tuning, "noise_cancellation_window");
  defaults.maxConsecutiveRuns = requirePositive(tuning, "max_consecutive_runs");
  if (tuning["maximum_executions"])
    defaults.maximumExecutions = requirePositive(tuning, "maximum_executions");
  if (tuning["maximum_retired_configurations"])
    defaults.maximumRetiredConfigurations =
        requirePositive(tuning, "maximum_retired_configurations");
  if (defaults.maxConsecutiveRuns <= defaults.warmupRuns)
    throw std::runtime_error{"YAML max_consecutive_runs must exceed "
                             "warmup_runs so every activation is measured."};
  if (persistence["file"] && persistence["directory"])
    throw std::runtime_error{
        "YAML persistence must define either file or directory, not both."};
  if (persistence["file"]) {
    auto const file = persistence["file"].as<std::string>();
    if (file.empty())
      throw std::runtime_error{"YAML persistence.file must not be empty."};
    defaults.persistenceFile = file;
  } else if (persistence["directory"]) {
    auto const directory = persistence["directory"].as<std::string>();
    if (directory.empty())
      throw std::runtime_error{"YAML persistence.directory must not be empty."};
    defaults.persistenceFile =
        std::filesystem::path{directory} / "history.json";
  } else {
    throw std::runtime_error{"YAML persistence.file is required."};
  }
  if (learning && learning["model"]) {
    auto const model = learning["model"].as<std::string>();
    if (model.empty())
      throw std::runtime_error{"YAML learning.model must not be empty."};
    defaults.learnedModelFile = model;
  }
  if (learning && learning["fallback"])
    defaults.learnedFallback =
        strategyFromName(learning["fallback"].as<std::string>());
  if (learning) {
    defaults.learnedCandidatePoolSize = optionalPositive(
        learning, "candidate_pool_size", defaults.learnedCandidatePoolSize);
    defaults.learnedCandidateBatchSize = optionalPositive(
        learning, "candidate_batch_size", defaults.learnedCandidateBatchSize);
  }
  defaults.validate();
  return defaults;
}

inline auto canonicalConfigPath(std::filesystem::path path)
    -> std::filesystem::path {
  if (path.empty())
    throw std::runtime_error{
        "The alpakaTune YAML configuration path must not be empty."};
  if (!std::filesystem::exists(path))
    throw std::runtime_error{"alpakaTune YAML configuration does not exist: " +
                             path.string()};
  return std::filesystem::weakly_canonical(std::move(path));
}

inline auto defaultConfigurationPath() -> std::filesystem::path {
  if (auto const *configured = std::getenv("ALPAKA_TUNE_CONFIG");
      configured != nullptr && configured[0] != '\0')
    return configured;
#ifdef ALPAKA_TUNE_DEFAULT_CONFIG
  return ALPAKA_TUNE_DEFAULT_CONFIG;
#else
  throw std::runtime_error{
      "No alpakaTune YAML configuration was found. Set ALPAKA_TUNE_CONFIG."};
#endif
}

} // namespace detail

inline void TunerConfig::validate() const {
  auto positive = [](std::size_t value, std::string_view member) {
    if (value == 0u)
      throw std::invalid_argument{std::string{member} +
                                  " must be greater than zero."};
  };
  auto positiveFinite = [](double value, std::string_view member) {
    if (!std::isfinite(value) || value <= 0.0)
      throw std::invalid_argument{std::string{member} +
                                  " must be positive and finite."};
  };

  positive(runsPerCandidate, "TunerConfig::runsPerCandidate");
  positive(minimumRunsPerCandidate, "TunerConfig::minimumRunsPerCandidate");
  if (minimumRunsPerCandidate > runsPerCandidate)
    throw std::invalid_argument{"TunerConfig::minimumRunsPerCandidate must not "
                                "exceed runsPerCandidate."};
  positive(ciCheckInterval, "TunerConfig::ciCheckInterval");
  positiveFinite(ciZScore, "TunerConfig::ciZScore");
  positiveFinite(ciRelativeWidth, "TunerConfig::ciRelativeWidth");
  positiveFinite(outlierMadScale, "TunerConfig::outlierMadScale");
  positive(mannWhitneyMinimumSamples, "TunerConfig::mannWhitneyMinimumSamples");
  if (!std::isfinite(mannWhitneyAlpha) || mannWhitneyAlpha <= 0.0 ||
      mannWhitneyAlpha >= 1.0)
    throw std::invalid_argument{
        "TunerConfig::mannWhitneyAlpha must be finite and in (0, 1)."};
  positive(noiseCancellationWindow, "TunerConfig::noiseCancellationWindow");
  positive(maxConsecutiveRuns, "TunerConfig::maxConsecutiveRuns");
  if (maxConsecutiveRuns <= warmupRuns)
    throw std::invalid_argument{
        "TunerConfig::maxConsecutiveRuns must exceed warmupRuns."};
  if (maximumExecutions)
    positive(*maximumExecutions, "TunerConfig::maximumExecutions");
  if (maximumRetiredConfigurations)
    positive(*maximumRetiredConfigurations,
             "TunerConfig::maximumRetiredConfigurations");
  if (persistenceFile.empty())
    throw std::invalid_argument{
        "TunerConfig::persistenceFile must not be empty."};
  if (learnedFallback != StrategyKind::random)
    throw std::invalid_argument{"TunerConfig::learnedFallback currently "
                                "supports only StrategyKind::random."};
  positive(learnedCandidatePoolSize,
           "TunerConfig::learnedCandidatePoolSize");
  positive(learnedCandidateBatchSize,
           "TunerConfig::learnedCandidateBatchSize");
  if (learnedCandidateBatchSize > learnedCandidatePoolSize)
    throw std::invalid_argument{
        "TunerConfig::learnedCandidateBatchSize must not exceed "
        "learnedCandidatePoolSize."};
}

inline auto TunerConfig::fromYaml(std::filesystem::path path) -> TunerConfig {
  return detail::loadTunerConfig(detail::canonicalConfigPath(std::move(path)));
}

[[nodiscard]] inline auto tunerConfig() -> TunerConfig {
  static TunerConfig const defaultConfig =
      TunerConfig::fromYaml(detail::defaultConfigurationPath());
  return defaultConfig;
}
[[nodiscard]] inline auto tunerConfig(std::filesystem::path path)
    -> TunerConfig {
  return TunerConfig::fromYaml(std::move(path));
}

} // namespace alpakaTune
