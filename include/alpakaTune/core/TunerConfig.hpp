// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "alpakaTune/space/SpaceConfig.hpp"

#include "alpakaTune/interfaces/Strategy.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace alpakaTune {

/** @brief Whether the tuner may search for new configurations. */
enum class ExplorationPolicy { online, offline };

/** @brief Whether selection keeps responding to measured performance. */
enum class SelectionPolicy { adaptive, fixed };

[[nodiscard]] constexpr auto explorationPolicyName(ExplorationPolicy policy)
    -> std::string_view {
  return policy == ExplorationPolicy::online ? "online" : "offline";
}

[[nodiscard]] constexpr auto selectionPolicyName(SelectionPolicy policy)
    -> std::string_view {
  return policy == SelectionPolicy::adaptive ? "adaptive" : "fixed";
}

[[nodiscard]] inline auto explorationPolicyFromName(std::string_view name)
    -> ExplorationPolicy {
  if (name == "online")
    return ExplorationPolicy::online;
  if (name == "offline")
    return ExplorationPolicy::offline;
  throw std::runtime_error{
      "YAML tuning.exploration must be online or offline."};
}

[[nodiscard]] inline auto selectionPolicyFromName(std::string_view name)
    -> SelectionPolicy {
  if (name == "adaptive")
    return SelectionPolicy::adaptive;
  if (name == "fixed")
    return SelectionPolicy::fixed;
  throw std::runtime_error{"YAML tuning.selection must be adaptive or fixed."};
}

/** @brief Access policy for the compact sampled history. */
struct HistoryConfig {
  /** Optional compact-history JSON file. */
  std::optional<std::filesystem::path> file;
  /** Load compatible compact contexts when the file exists. */
  bool read{true};
  /** Replace or update the compact file during normal process shutdown. */
  bool write{true};
  /** Maximum configurations written per context; omission writes all. */
  std::optional<std::size_t> sampleCount;
};

/** @brief Access policy for the complete raw-sample history. */
struct CompleteHistoryConfig {
  /** Optional complete-history JSON file. */
  std::optional<std::filesystem::path> file;
  /** Load compatible complete contexts when the file exists. */
  bool read{true};
  /** Replace or update the complete file during normal process shutdown. */
  bool write{true};
};

/** @brief Optional active-candidate scheduler configuration. */
struct QueueConfig {
  /** Explicitly bypass the queue while retaining its saved parameters. */
  bool disable{false};
  /** Untimed launches at the beginning of every queue activation. */
  std::size_t warmupRuns{1u};
  /** Maximum number of distinct candidates interleaved in the active queue. */
  std::size_t noiseCancellationWindow{50u};
  /** Launches in one queue activation, including warm-up launches. */
  std::size_t maxConsecutiveRuns{3u};
};

/** @brief Complete, copyable policy used to construct one or more tuners.
 *
 * A tuner snapshots this aggregate at construction. Configuration controls
 * tuner policy only; it never determines the surrounding application's loop.
 */
struct TunerConfig {
  SpaceConfig space;
  /** Permission to search for configurations not yet measured. */
  ExplorationPolicy exploration{ExplorationPolicy::online};
  /** Selection behavior after exploration ends. */
  SelectionPolicy selection{SelectionPolicy::adaptive};
  /** Successful reuse launches between probes of known alternatives. */
  std::size_t adaptiveProbeInterval{10u};
  /** Launch a terminal replay winner without runtime instrumentation. */
  bool replayFastPath{false};
  /** Optional active-candidate scheduler; omission launches directly. */
  std::optional<QueueConfig> queue;
  /** Maximum new-run measurements per configuration in online exploration with
   * fixed selection. */
  std::size_t runsPerCandidate{1u};
  /** New-run measurements required before fixed-mode CI retirement. */
  std::size_t minimumRunsPerCandidate{1u};
  /** Recompute the fixed-mode confidence interval at this sample cadence. */
  std::size_t ciCheckInterval{10u};
  /** Z score used by the non-parametric median confidence interval. */
  double ciZScore{2.576};
  /** Fixed-mode retirement threshold for relative CI width. */
  double ciRelativeWidth{0.05};
  /** Median-absolute-deviation multiplier used to reject decision outliers. */
  double outlierMadScale{3.5};
  /** Enable fixed-mode early retirement of statistically slower candidates. */
  bool mannWhitneyEarlyStop{true};
  /** Samples required in both records before the rank test is eligible. */
  std::size_t mannWhitneyMinimumSamples{8u};
  /** One-sided significance level for Mann-Whitney retirement. */
  double mannWhitneyAlpha{0.05};
  /** Consecutive rejected proposals allowed in one admission attempt. */
  std::size_t maximumConsecutiveStrategyRetries{20u};
  /** Online exploration completion guard on launches in the current online run.
   */
  std::optional<std::size_t> maximumExecutions;
  /** Alternative online-fixed completion guard on retired configurations. */
  std::optional<std::size_t> maximumRetiredConfigurations;
  /** Optional new-run launch horizon used by online exploration with adaptive
   * selection. */
  std::optional<std::size_t> horizon;
  /** Maximum number of newest timing records retained per configuration. */
  std::size_t historyWindowSize{10u};
  /** Shape of the normalized-logistic adaptive revisit-admission ramp. */
  double revisitAdmissionSteepness{16.0};
  /** Relative-Boltzmann temperature at execution zero. */
  double scoreTemperatureStart{0.25};
  /** Relative-Boltzmann temperature at the adaptive execution horizon. */
  double scoreTemperatureEnd{0.05};
  /** Initial adaptive progress when compatible measured history is active. */
  double horizonOffsetWithActiveHistory{0.8};
  /** Parameter-proposal algorithm; admission remains tuner-owned. */
  StrategyKind strategy{StrategyKind::exhaustive};
  /** Base seed; disengage for a nondeterministic process-local seed. */
  std::optional<std::uint64_t> randomSeed{0u};
  /** Compact sampled history policy. */
  HistoryConfig history;
  /** Complete raw-sample history policy. */
  CompleteHistoryConfig completeHistory;
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

  /** @brief Load a mutable configuration from schema-version-4 YAML.
   * @throws std::runtime_error for missing, malformed, or unsupported input.
   */
  [[nodiscard]] static auto fromYaml(std::filesystem::path path) -> TunerConfig;

  /** @brief Reject values which cannot form a valid tuner.
   * @throws std::invalid_argument when any cross-field invariant is violated.
   */
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

inline auto optionalUnitInterval(YAML::Node const &node, char const *key,
                                 double fallback) -> double {
  if (!node[key])
    return fallback;
  auto const value = node[key].as<double>();
  if (!std::isfinite(value) || value < 0.0 || value > 1.0)
    throw std::runtime_error{
        std::string{"YAML key must be finite and in [0, 1]: "} + key};
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
  rejectUnknown(root,
                {"schema_version", "tuning", "queue", "history",
                 "complete_history", "learning", "space"},
                "root");
  if (!root["schema_version"])
    throw std::runtime_error{"Missing YAML key: schema_version"};
  auto const schemaVersion = root["schema_version"].as<int>();
  if (schemaVersion != 4)
    throw std::runtime_error{"Unsupported alpakaTune YAML schema_version; "
                             "migrate to schema_version: "
                             "4 with tuning.exploration and tuning.selection "
                             "instead of tuning.mode."};
  if (!root["tuning"] || !root["tuning"].IsMap())
    throw std::runtime_error{"Missing YAML map: tuning"};
  if (root["queue"] && !root["queue"].IsMap())
    throw std::runtime_error{"YAML queue must contain a map."};
  if (root["history"] && !root["history"].IsMap())
    throw std::runtime_error{"YAML history must contain a map."};
  if (root["complete_history"] && !root["complete_history"].IsMap())
    throw std::runtime_error{"YAML complete_history must contain a map."};

  auto const tuning = root["tuning"];
  auto const queue = root["queue"];
  auto const history = root["history"];
  auto const completeHistory = root["complete_history"];
  auto const learning = root["learning"];
  if (learning && !learning.IsMap())
    throw std::runtime_error{"YAML learning must contain a map."};
  rejectUnknown(tuning,
                {"exploration",
                 "selection",
                 "adaptive_probe_interval",
                 "replay_fast_path",
                 "strategy",
                 "random_seed",
                 "warmup_runs",
                 "runs_per_candidate",
                 "minimum_runs_per_candidate",
                 "ci_check_interval",
                 "ci_z_score",
                 "ci_relative_width",
                 "outlier_mad_scale",
                 "mann_whitney_early_stop",
                 "mann_whitney_min_samples",
                 "mann_whitney_alpha",
                 "noise_cancellation_window",
                 "max_consecutive_runs",
                 "maximum_consecutive_strategy_retries",
                 "horizon",
                 "maximum_executions",
                 "maximum_retired_configurations",
                 "history_window_size",
                 "revisit_admission_steepness",
                 "score_temperature_start",
                 "score_temperature_end",
                 "horizon_offset_with_active_history"},
                "tuning");
  if (queue)
    rejectUnknown(queue,
                  {"disable", "warmup_runs", "noise_cancellation_window",
                   "max_consecutive_runs"},
                  "queue");
  if (history)
    rejectUnknown(history, {"file", "read", "write", "sample_count"},
                  "history");
  if (completeHistory)
    rejectUnknown(completeHistory, {"file", "read", "write"},
                  "complete_history");
  if (learning)
    rejectUnknown(
        learning,
        {"model", "fallback", "candidate_pool_size", "candidate_batch_size"},
        "learning");

  TunerConfig defaults;
  if (tuning["exploration"])
    defaults.exploration =
        explorationPolicyFromName(tuning["exploration"].as<std::string>());
  if (tuning["selection"])
    defaults.selection =
        selectionPolicyFromName(tuning["selection"].as<std::string>());
  defaults.adaptiveProbeInterval = optionalPositive(
      tuning, "adaptive_probe_interval", defaults.adaptiveProbeInterval);
  defaults.replayFastPath = tuning["replay_fast_path"]
                                ? tuning["replay_fast_path"].as<bool>()
                                : defaults.replayFastPath;
  if (queue) {
    defaults.queue.emplace();
    defaults.queue->disable =
        queue["disable"] ? queue["disable"].as<bool>() : false;
    if (queue["warmup_runs"])
      defaults.queue->warmupRuns = queue["warmup_runs"].as<std::size_t>();
    if (queue["noise_cancellation_window"])
      defaults.queue->noiseCancellationWindow =
          queue["noise_cancellation_window"].as<std::size_t>();
    if (queue["max_consecutive_runs"])
      defaults.queue->maxConsecutiveRuns =
          queue["max_consecutive_runs"].as<std::size_t>();
  }
  auto const strategy =
      tuning["strategy"] ? tuning["strategy"].as<std::string>() : "exhaustive";
  defaults.strategy = strategyFromName(strategy);
  if (auto const seed = tuning["random_seed"]; seed.IsDefined()) {
    if (seed.IsNull())
      throw std::runtime_error{
          "YAML random_seed must be an unsigned integer or "
          "'nondeterministic'."};
    auto const spelling = seed.as<std::string>();
    if (spelling == "nondeterministic")
      defaults.randomSeed.reset();
    else {
      try {
        defaults.randomSeed = seed.as<std::uint64_t>();
      } catch (YAML::Exception const &) {
        throw std::runtime_error{
            "YAML random_seed must be an unsigned integer or "
            "'nondeterministic'."};
      }
    }
  }
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
  defaults.maximumConsecutiveStrategyRetries =
      optionalPositive(tuning, "maximum_consecutive_strategy_retries",
                       defaults.maximumConsecutiveStrategyRetries);
  if (auto const limit = tuning["maximum_executions"]; limit.IsDefined()) {
    if (limit.IsNull())
      defaults.maximumExecutions.reset();
    else
      defaults.maximumExecutions =
          requirePositive(tuning, "maximum_executions");
  }
  if (auto const limit = tuning["maximum_retired_configurations"];
      limit.IsDefined()) {
    if (limit.IsNull())
      defaults.maximumRetiredConfigurations.reset();
    else
      defaults.maximumRetiredConfigurations =
          requirePositive(tuning, "maximum_retired_configurations");
  }
  if (auto const horizon = tuning["horizon"]; horizon.IsDefined()) {
    if (horizon.IsNull())
      defaults.horizon.reset();
    else
      defaults.horizon = requirePositive(tuning, "horizon");
  }
  if (!defaults.horizon &&
      tuning["horizon_offset_with_active_history"].IsDefined())
    throw std::runtime_error{
        "YAML horizon_offset_with_active_history requires horizon."};
  defaults.historyWindowSize = optionalPositive(tuning, "history_window_size",
                                                defaults.historyWindowSize);
  defaults.revisitAdmissionSteepness =
      optionalPositiveFinite(tuning, "revisit_admission_steepness",
                             defaults.revisitAdmissionSteepness);
  defaults.scoreTemperatureStart = optionalPositiveFinite(
      tuning, "score_temperature_start", defaults.scoreTemperatureStart);
  defaults.scoreTemperatureEnd = optionalPositiveFinite(
      tuning, "score_temperature_end", defaults.scoreTemperatureEnd);
  defaults.horizonOffsetWithActiveHistory =
      optionalUnitInterval(tuning, "horizon_offset_with_active_history",
                           defaults.horizonOffsetWithActiveHistory);
  if (history && history["file"]) {
    auto const file = history["file"].as<std::string>();
    if (file.empty())
      throw std::runtime_error{"YAML history.file must not be empty."};
    defaults.history.file = file;
  }
  if (history && history["read"])
    defaults.history.read = history["read"].as<bool>();
  if (history && history["write"])
    defaults.history.write = history["write"].as<bool>();
  if (history && history["sample_count"])
    defaults.history.sampleCount = requirePositive(history, "sample_count");
  if (completeHistory && completeHistory["file"]) {
    auto const file = completeHistory["file"].as<std::string>();
    if (file.empty())
      throw std::runtime_error{"YAML complete_history.file must not be empty."};
    defaults.completeHistory.file = file;
  }
  if (completeHistory && completeHistory["read"])
    defaults.completeHistory.read = completeHistory["read"].as<bool>();
  if (completeHistory && completeHistory["write"])
    defaults.completeHistory.write = completeHistory["write"].as<bool>();
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
  if (auto const space = root["space"]; space) {
    if (!space.IsMap())
      throw std::runtime_error{"YAML space must contain a map."};
    rejectUnknown(space,
                  {"initial_candidates", "refinement_batch_size",
                   "refinement_interval", "maximum_candidates",
                   "maximum_generation_attempts", "plateau_patience",
                   "minimum_relative_improvement"},
                  "space");
    defaults.space.initialCandidates = optionalPositive(
        space, "initial_candidates", defaults.space.initialCandidates);
    defaults.space.refinementBatchSize = optionalPositive(
        space, "refinement_batch_size", defaults.space.refinementBatchSize);
    defaults.space.refinementInterval = optionalPositive(
        space, "refinement_interval", defaults.space.refinementInterval);
    defaults.space.maximumCandidates = optionalPositive(
        space, "maximum_candidates", defaults.space.maximumCandidates);
    defaults.space.maximumGenerationAttempts =
        optionalPositive(space, "maximum_generation_attempts",
                         defaults.space.maximumGenerationAttempts);
    if (space["plateau_patience"] && !space["plateau_patience"].IsNull())
      defaults.space.plateauPatience =
          requirePositive(space, "plateau_patience");
    if (space["minimum_relative_improvement"])
      defaults.space.minimumRelativeImprovement =
          space["minimum_relative_improvement"].as<double>();
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
  space.validate();
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
  if (queue && !queue->disable) {
    positive(queue->noiseCancellationWindow,
             "TunerConfig::queue.noiseCancellationWindow");
    positive(queue->maxConsecutiveRuns,
             "TunerConfig::queue.maxConsecutiveRuns");
    if (queue->maxConsecutiveRuns <= queue->warmupRuns)
      throw std::invalid_argument{
          "TunerConfig::queue.maxConsecutiveRuns must exceed "
          "queue.warmupRuns."};
  }
  positive(maximumConsecutiveStrategyRetries,
           "TunerConfig::maximumConsecutiveStrategyRetries");
  positive(historyWindowSize, "TunerConfig::historyWindowSize");
  positiveFinite(revisitAdmissionSteepness,
                 "TunerConfig::revisitAdmissionSteepness");
  positiveFinite(scoreTemperatureStart, "TunerConfig::scoreTemperatureStart");
  positiveFinite(scoreTemperatureEnd, "TunerConfig::scoreTemperatureEnd");
  if (!std::isfinite(horizonOffsetWithActiveHistory) ||
      horizonOffsetWithActiveHistory < 0.0 ||
      horizonOffsetWithActiveHistory > 1.0)
    throw std::invalid_argument{
        "TunerConfig::horizonOffsetWithActiveHistory must be finite and in "
        "[0, 1]."};
  if (scoreTemperatureEnd > scoreTemperatureStart)
    throw std::invalid_argument{
        "TunerConfig::scoreTemperatureEnd must not exceed "
        "scoreTemperatureStart."};
  if (maximumExecutions)
    positive(*maximumExecutions, "TunerConfig::maximumExecutions");
  if (maximumRetiredConfigurations)
    positive(*maximumRetiredConfigurations,
             "TunerConfig::maximumRetiredConfigurations");
  positive(adaptiveProbeInterval, "TunerConfig::adaptiveProbeInterval");
  if (exploration == ExplorationPolicy::online &&
      selection == SelectionPolicy::fixed && !maximumExecutions &&
      !maximumRetiredConfigurations)
    throw std::invalid_argument{
        "Online fixed selection requires maximumExecutions or "
        "maximumRetiredConfigurations."};
  if (selection == SelectionPolicy::adaptive && replayFastPath)
    throw std::invalid_argument{
        "TunerConfig::replayFastPath requires fixed selection."};
  if (horizon) {
    positive(*horizon, "TunerConfig::horizon");
    if (exploration != ExplorationPolicy::online ||
        selection != SelectionPolicy::adaptive)
      throw std::invalid_argument{"TunerConfig::horizon requires online "
                                  "exploration and adaptive selection."};
  }
  if (exploration == ExplorationPolicy::offline &&
      (maximumExecutions || maximumRetiredConfigurations))
    throw std::invalid_argument{
        "Exploration completion guards require online exploration."};
  if (selection == SelectionPolicy::fixed &&
      exploration == ExplorationPolicy::online &&
      runsPerCandidate > historyWindowSize)
    throw std::invalid_argument{
        "TunerConfig::runsPerCandidate must not exceed historyWindowSize with "
        "online fixed selection."};
  if (history.file && history.file->empty())
    throw std::invalid_argument{
        "TunerConfig::history.file must contain a non-empty path."};
  if (completeHistory.file && completeHistory.file->empty())
    throw std::invalid_argument{
        "TunerConfig::completeHistory.file must contain a non-empty path."};
  if (history.sampleCount)
    positive(*history.sampleCount, "TunerConfig::history.sampleCount");
  if (history.file && completeHistory.file &&
      std::filesystem::absolute(*history.file).lexically_normal() ==
          std::filesystem::absolute(*completeHistory.file).lexically_normal())
    throw std::invalid_argument{
        "Compact and complete history must use different files."};
  if (learnedFallback != StrategyKind::random)
    throw std::invalid_argument{"TunerConfig::learnedFallback currently "
                                "supports only StrategyKind::random."};
  positive(learnedCandidatePoolSize, "TunerConfig::learnedCandidatePoolSize");
  positive(learnedCandidateBatchSize, "TunerConfig::learnedCandidateBatchSize");
  if (learnedCandidateBatchSize > learnedCandidatePoolSize)
    throw std::invalid_argument{
        "TunerConfig::learnedCandidateBatchSize must not exceed "
        "learnedCandidatePoolSize."};
}

inline auto TunerConfig::fromYaml(std::filesystem::path path) -> TunerConfig {
  return detail::loadTunerConfig(detail::canonicalConfigPath(std::move(path)));
}

/** @brief Return a mutable copy of the process-default YAML configuration.
 *
 * The source path is selected once from ALPAKA_TUNE_CONFIG or the installed
 * default. Each returned value can be modified independently before makeTuner.
 */
[[nodiscard]] inline auto tunerConfig() -> TunerConfig {
  static TunerConfig const defaultConfig =
      TunerConfig::fromYaml(detail::defaultConfigurationPath());
  return defaultConfig;
}
/** @brief Load a mutable configuration from an explicit YAML path. */
[[nodiscard]] inline auto tunerConfig(std::filesystem::path path)
    -> TunerConfig {
  return TunerConfig::fromYaml(std::move(path));
}

} // namespace alpakaTune
