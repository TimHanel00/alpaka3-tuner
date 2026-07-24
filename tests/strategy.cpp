// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#include <alpakaTune/alpakaTune.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace {

class TestStrategyContext final : public alpakaTune::StrategyContext {
public:
  [[nodiscard]] auto parameterSizes() const noexcept
      -> std::span<std::size_t const> override {
    return m_sizes;
  }

  [[nodiscard]] auto
  runtimeFor(alpakaTune::ParameterConfiguration const &configuration) const
      -> std::optional<alpakaTune::RuntimeObservation> override {
    for (auto const &[knownConfiguration, runtime] : m_runtimes) {
      if (knownConfiguration == configuration)
        return alpakaTune::RuntimeObservation{
            runtime,
            8u,
            8u,
            alpakaTune::ConfigurationState::retired,
            alpakaTune::RuntimeComparison::inconclusive,
            true};
    }
    return std::nullopt;
  }

  void record(alpakaTune::ParameterConfiguration configuration,
              double runtime) {
    m_runtimes.emplace_back(std::move(configuration), runtime);
  }

private:
  std::array<std::size_t, 3u> m_sizes{2u, 3u, 4u};
  std::vector<std::pair<alpakaTune::ParameterConfiguration, double>> m_runtimes;
};

class DiversityStrategyContext final : public alpakaTune::StrategyContext {
public:
  [[nodiscard]] auto parameterSizes() const noexcept
      -> std::span<std::size_t const> override {
    return m_sizes;
  }

  [[nodiscard]] auto
  runtimeFor(alpakaTune::ParameterConfiguration const &) const
      -> std::optional<alpakaTune::RuntimeObservation> override {
    return std::nullopt;
  }

private:
  std::array<std::size_t, 1u> m_sizes{100u};
};

auto valid(alpakaTune::ParameterConfiguration const &configuration) -> bool {
  if (configuration.size() != 3u)
    return false;
  for (auto const value : configuration) {
    if (!std::isfinite(value) || value < 0.0f || value > 1.0f)
      return false;
  }
  return true;
}

} // namespace

auto main() -> int {
  auto defaultConfig = alpakaTune::TunerConfig{};
  defaultConfig.validate();
  if (defaultConfig.mode != alpakaTune::TuningMode::onlineAdaptive ||
      defaultConfig.maximumExecutions ||
      defaultConfig.maximumRetiredConfigurations ||
      defaultConfig.maximumConsecutiveStrategyRetries != 20u ||
      defaultConfig.horizon != 40'000u || defaultConfig.history.file ||
      !defaultConfig.history.read || !defaultConfig.history.write ||
      defaultConfig.history.sampleCount || defaultConfig.completeHistory.file ||
      !defaultConfig.completeHistory.read ||
      !defaultConfig.completeHistory.write ||
      std::abs(defaultConfig.horizonOffsetWithActiveHistory - 0.8) > 1.0e-12)
    return EXIT_FAILURE;
  for (auto const invalidOffset : {-0.01, 1.01}) {
    auto invalidConfig = defaultConfig;
    invalidConfig.horizonOffsetWithActiveHistory = invalidOffset;
    auto rejected = false;
    try {
      invalidConfig.validate();
    } catch (std::invalid_argument const &) {
      rejected = true;
    }
    if (!rejected)
      return EXIT_FAILURE;
  }
  auto mixedAdaptiveConfig = defaultConfig;
  mixedAdaptiveConfig.maximumExecutions = 10u;
  auto mixedAdaptiveConfigRejected = false;
  try {
    mixedAdaptiveConfig.validate();
  } catch (std::invalid_argument const &) {
    mixedAdaptiveConfigRejected = true;
  }
  if (!mixedAdaptiveConfigRejected)
    return EXIT_FAILURE;
  auto invalidRetryConfig = defaultConfig;
  invalidRetryConfig.maximumConsecutiveStrategyRetries = 0u;
  auto invalidRetryConfigRejected = false;
  try {
    invalidRetryConfig.validate();
  } catch (std::invalid_argument const &) {
    invalidRetryConfigRejected = true;
  }
  if (!invalidRetryConfigRejected)
    return EXIT_FAILURE;

  for (auto const strategyKind :
       {alpakaTune::StrategyKind::random,
        alpakaTune::StrategyKind::simulatedAnnealing,
        alpakaTune::StrategyKind::bayesianOptimization}) {
    TestStrategyContext context;
    auto strategy = alpakaTune::makeParameterStrategy(strategyKind, 17u);
    auto const first = strategy->recommend(context);
    if (!valid(first) || context.runtimeFor(first).has_value())
      return EXIT_FAILURE;
    context.record(first, 1.0e-3);
    auto const second = strategy->recommend(context);
    if (!valid(second))
      return EXIT_FAILURE;
  }

  // Diversity is a strategy contract, not an admission-guard side effect.
  // Inspect raw recommendations without reporting them as scheduled.
  for (auto const strategyKind :
       {alpakaTune::StrategyKind::exhaustive, alpakaTune::StrategyKind::random,
        alpakaTune::StrategyKind::simulatedAnnealing,
        alpakaTune::StrategyKind::bayesianOptimization}) {
    constexpr auto deterministicContractSeed = std::uint64_t{0x5eedu};
    auto context = DiversityStrategyContext{};
    auto strategy = alpakaTune::makeParameterStrategy(
        strategyKind, deterministicContractSeed);
    auto repeated = alpakaTune::makeParameterStrategy(
        strategyKind, deterministicContractSeed);
    auto candidates = std::set<std::size_t>{};
    for (std::size_t recommendation = 0u; recommendation < 100u;
         ++recommendation) {
      auto const configuration = strategy->recommend(context);
      if (configuration.size() != 1u ||
          configuration != repeated->recommend(context))
        return EXIT_FAILURE;
      candidates.insert(static_cast<std::size_t>(
          std::lround(static_cast<double>(configuration.front()) * 99.0)));
    }
    if (candidates.size() < 10u)
      return EXIT_FAILURE;
  }

  if (alpakaTune::strategyFromName("exhaustive") !=
          alpakaTune::StrategyKind::exhaustive ||
      alpakaTune::strategyFromName("random") !=
          alpakaTune::StrategyKind::random ||
      alpakaTune::strategyFromName("simulated_annealing") !=
          alpakaTune::StrategyKind::simulatedAnnealing ||
      alpakaTune::strategyFromName("bayesian_optimization") !=
          alpakaTune::StrategyKind::bayesianOptimization ||
      alpakaTune::strategyFromName("learned_hybrid") !=
          alpakaTune::StrategyKind::learnedHybrid)
    return EXIT_FAILURE;

  auto const configurationDirectory =
      std::filesystem::temp_directory_path() / "alpakaTune-strategy-yaml-test";
  std::filesystem::remove_all(configurationDirectory);
  std::filesystem::create_directories(configurationDirectory);
  for (auto const &[name, expected] :
       std::array<std::pair<char const *, alpakaTune::StrategyKind>, 5u>{
           {{"exhaustive", alpakaTune::StrategyKind::exhaustive},
            {"random", alpakaTune::StrategyKind::random},
            {"simulated_annealing",
             alpakaTune::StrategyKind::simulatedAnnealing},
            {"bayesian_optimization",
             alpakaTune::StrategyKind::bayesianOptimization},
            {"learned_hybrid", alpakaTune::StrategyKind::learnedHybrid}}}) {
    auto const configuration =
        configurationDirectory / (std::string{name} + ".yaml");
    auto yaml = std::ofstream{configuration};
    yaml << "schema_version: 3\n"
            "tuning:\n"
            "  mode: online_fixed\n"
            "  strategy: "
         << name
         << "\n"
            "  random_seed: 0\n"
            "  warmup_runs: 0\n"
            "  runs_per_candidate: 1\n"
            "  minimum_runs_per_candidate: 1\n"
            "  ci_check_interval: 2\n"
            "  ci_z_score: 1.96\n"
            "  ci_relative_width: 0.1\n"
            "  outlier_mad_scale: 4.0\n"
            "  mann_whitney_early_stop: false\n"
            "  mann_whitney_min_samples: 8\n"
            "  mann_whitney_alpha: 0.1\n"
            "  noise_cancellation_window: 1\n"
            "  max_consecutive_runs: 1\n"
            "  maximum_executions: 100000\n"
            "  maximum_retired_configurations: 90000\n"
            "history:\n"
            "  file: .alpakaTune/history.json\n"
            "  sample_count: 7\n"
            "complete_history:\n"
            "  file: .alpakaTune/complete-history.json\n";
    yaml.close();
    auto const defaults = alpakaTune::TunerConfig::fromYaml(configuration);
    if (defaults.strategy != expected ||
        defaults.history.file != ".alpakaTune/history.json" ||
        defaults.history.sampleCount != 7u ||
        defaults.completeHistory.file != ".alpakaTune/complete-history.json" ||
        !defaults.history.read || !defaults.history.write ||
        !defaults.completeHistory.read || !defaults.completeHistory.write ||
        defaults.minimumRunsPerCandidate != 1u ||
        defaults.ciCheckInterval != 2u ||
        std::abs(defaults.ciZScore - 1.96) > 1.0e-12 ||
        std::abs(defaults.ciRelativeWidth - 0.1) > 1.0e-12 ||
        std::abs(defaults.outlierMadScale - 4.0) > 1.0e-12 ||
        defaults.mannWhitneyEarlyStop ||
        defaults.mannWhitneyMinimumSamples != 8u ||
        std::abs(defaults.mannWhitneyAlpha - 0.1) > 1.0e-12 ||
        defaults.maximumExecutions != 100000u ||
        defaults.maximumRetiredConfigurations != 90000u)
      return EXIT_FAILURE;
  }

  auto const memoryConfiguration = configurationDirectory / "memory-only.yaml";
  auto memoryYaml = std::ofstream{memoryConfiguration};
  memoryYaml << "schema_version: 2\n"
                "tuning:\n"
                "  mode: online_adaptive\n"
                "  strategy: random\n"
                "  warmup_runs: 0\n"
                "  runs_per_candidate: 1\n"
                "  noise_cancellation_window: 1\n"
                "  max_consecutive_runs: 1\n"
                "  horizon: 100\n";
  memoryYaml.close();
  auto const memoryDefaults =
      alpakaTune::TunerConfig::fromYaml(memoryConfiguration);
  if (memoryDefaults.history.file || !memoryDefaults.history.read ||
      !memoryDefaults.history.write || memoryDefaults.completeHistory.file ||
      !memoryDefaults.completeHistory.read ||
      !memoryDefaults.completeHistory.write)
    return EXIT_FAILURE;

  auto const processLocalConfiguration =
      configurationDirectory / "process-local.yaml";
  auto processLocalYaml = std::ofstream{processLocalConfiguration};
  processLocalYaml << "schema_version: 3\n"
                      "tuning:\n"
                      "  mode: online_adaptive\n"
                      "  strategy: random\n"
                      "  warmup_runs: 0\n"
                      "  runs_per_candidate: 1\n"
                      "  noise_cancellation_window: 1\n"
                      "  max_consecutive_runs: 1\n"
                      "  horizon: 100\n"
                      "history:\n"
                      "  read: false\n"
                      "  write: false\n"
                      "complete_history:\n"
                      "  read: false\n"
                      "  write: false\n";
  processLocalYaml.close();
  auto const processLocalDefaults =
      alpakaTune::TunerConfig::fromYaml(processLocalConfiguration);
  if (processLocalDefaults.history.file || processLocalDefaults.history.read ||
      processLocalDefaults.history.write ||
      processLocalDefaults.completeHistory.file ||
      processLocalDefaults.completeHistory.read ||
      processLocalDefaults.completeHistory.write)
    return EXIT_FAILURE;

  auto const accessConfiguration =
      configurationDirectory / "history-access.yaml";
  auto accessYaml = std::ofstream{accessConfiguration};
  accessYaml << "schema_version: 3\n"
                "tuning:\n"
                "  mode: offline\n"
                "  strategy: random\n"
                "  warmup_runs: 0\n"
                "  runs_per_candidate: 1\n"
                "  noise_cancellation_window: 1\n"
                "  max_consecutive_runs: 1\n"
                "history:\n"
                "  file: read-only-history.json\n"
                "  read: true\n"
                "  write: false\n"
                "  sample_count: 5\n"
                "complete_history:\n"
                "  file: complete.json\n"
                "  read: false\n"
                "  write: true\n";
  accessYaml.close();
  auto const accessDefaults =
      alpakaTune::TunerConfig::fromYaml(accessConfiguration);
  if (accessDefaults.history.file != "read-only-history.json" ||
      !accessDefaults.history.read || accessDefaults.history.write ||
      accessDefaults.history.sampleCount != 5u ||
      accessDefaults.completeHistory.file != "complete.json" ||
      accessDefaults.completeHistory.read ||
      !accessDefaults.completeHistory.write)
    return EXIT_FAILURE;

  auto const writeOnlyConfiguration =
      configurationDirectory / "write-only-history.yaml";
  auto writeOnlyYaml = std::ofstream{writeOnlyConfiguration};
  writeOnlyYaml << "schema_version: 3\n"
                   "tuning:\n"
                   "  mode: online_fixed\n"
                   "  strategy: random\n"
                   "  warmup_runs: 0\n"
                   "  runs_per_candidate: 1\n"
                   "  noise_cancellation_window: 1\n"
                   "  max_consecutive_runs: 1\n"
                   "  maximum_executions: 100\n"
                   "complete_history:\n"
                   "  file: fresh-complete-history.json\n"
                   "  read: false\n"
                   "  write: true\n";
  writeOnlyYaml.close();
  auto const writeOnlyDefaults =
      alpakaTune::TunerConfig::fromYaml(writeOnlyConfiguration);
  if (writeOnlyDefaults.completeHistory.file != "fresh-complete-history.json" ||
      writeOnlyDefaults.completeHistory.read ||
      !writeOnlyDefaults.completeHistory.write)
    return EXIT_FAILURE;

  auto const learnedConfiguration = configurationDirectory / "learned-v3.yaml";
  auto learnedYaml = std::ofstream{learnedConfiguration};
  learnedYaml << "schema_version: 3\n"
                 "tuning:\n"
                 "  mode: online_fixed\n"
                 "  strategy: learned_hybrid\n"
                 "  warmup_runs: 0\n"
                 "  runs_per_candidate: 1\n"
                 "  noise_cancellation_window: 1\n"
                 "  max_consecutive_runs: 1\n"
                 "  maximum_executions: 100000\n"
                 "history:\n"
                 "  file: learned-history.json\n"
                 "learning:\n"
                 "  model: alternate.atml\n"
                 "  fallback: random\n"
                 "  candidate_pool_size: 64\n"
                 "  candidate_batch_size: 16\n";
  learnedYaml.close();
  auto const learnedDefaults =
      alpakaTune::TunerConfig::fromYaml(learnedConfiguration);
  if (learnedDefaults.strategy != alpakaTune::StrategyKind::learnedHybrid ||
      learnedDefaults.learnedModelFile != "alternate.atml" ||
      learnedDefaults.learnedFallback != alpakaTune::StrategyKind::random ||
      learnedDefaults.learnedCandidatePoolSize != 64u ||
      learnedDefaults.learnedCandidateBatchSize != 16u)
    return EXIT_FAILURE;

  auto const adaptiveConfiguration =
      configurationDirectory / "adaptive-v3.yaml";
  auto adaptiveYaml = std::ofstream{adaptiveConfiguration};
  adaptiveYaml << "schema_version: 3\n"
                  "tuning:\n"
                  "  mode: online_adaptive\n"
                  "  strategy: random\n"
                  "  warmup_runs: 1\n"
                  "  runs_per_candidate: 1\n"
                  "  noise_cancellation_window: 2\n"
                  "  max_consecutive_runs: 4\n"
                  "  maximum_consecutive_strategy_retries: 7\n"
                  "  horizon: 4000\n"
                  "  history_window_size: 10\n"
                  "  revisit_admission_steepness: 16\n"
                  "  score_temperature_start: 0.25\n"
                  "  score_temperature_end: 0.05\n"
                  "  horizon_offset_with_active_history: 0.7\n"
                  "history:\n"
                  "  file: adaptive-history.json\n";
  adaptiveYaml.close();
  auto const adaptiveDefaults =
      alpakaTune::TunerConfig::fromYaml(adaptiveConfiguration);
  if (adaptiveDefaults.mode != alpakaTune::TuningMode::onlineAdaptive ||
      adaptiveDefaults.horizon != 4000u || adaptiveDefaults.maximumExecutions ||
      adaptiveDefaults.maximumRetiredConfigurations ||
      adaptiveDefaults.maximumConsecutiveStrategyRetries != 7u ||
      adaptiveDefaults.historyWindowSize != 10u ||
      std::abs(adaptiveDefaults.revisitAdmissionSteepness - 16.0) > 1.0e-12 ||
      std::abs(adaptiveDefaults.scoreTemperatureStart - 0.25) > 1.0e-12 ||
      std::abs(adaptiveDefaults.scoreTemperatureEnd - 0.05) > 1.0e-12 ||
      std::abs(adaptiveDefaults.horizonOffsetWithActiveHistory - 0.7) > 1.0e-12)
    return EXIT_FAILURE;

  auto const mixedAdaptiveConfiguration =
      configurationDirectory / "mixed-adaptive-v2.yaml";
  auto mixedAdaptiveYaml = std::ofstream{mixedAdaptiveConfiguration};
  mixedAdaptiveYaml << "schema_version: 2\n"
                       "tuning:\n"
                       "  mode: online_adaptive\n"
                       "  strategy: random\n"
                       "  warmup_runs: 0\n"
                       "  runs_per_candidate: 1\n"
                       "  noise_cancellation_window: 1\n"
                       "  max_consecutive_runs: 1\n"
                       "  horizon: 100\n"
                       "  maximum_executions: 100\n";
  mixedAdaptiveYaml.close();
  auto mixedAdaptiveRejected = false;
  try {
    static_cast<void>(
        alpakaTune::TunerConfig::fromYaml(mixedAdaptiveConfiguration));
  } catch (std::runtime_error const &) {
    mixedAdaptiveRejected = true;
  }
  if (!mixedAdaptiveRejected)
    return EXIT_FAILURE;

  auto const mixedFixedConfiguration =
      configurationDirectory / "mixed-fixed-v2.yaml";
  auto mixedFixedYaml = std::ofstream{mixedFixedConfiguration};
  mixedFixedYaml << "schema_version: 2\n"
                    "tuning:\n"
                    "  mode: online_fixed\n"
                    "  strategy: random\n"
                    "  warmup_runs: 0\n"
                    "  runs_per_candidate: 1\n"
                    "  noise_cancellation_window: 1\n"
                    "  max_consecutive_runs: 1\n"
                    "  horizon: 100\n"
                    "  maximum_executions: 100\n";
  mixedFixedYaml.close();
  auto mixedFixedRejected = false;
  try {
    static_cast<void>(
        alpakaTune::TunerConfig::fromYaml(mixedFixedConfiguration));
  } catch (std::runtime_error const &) {
    mixedFixedRejected = true;
  }
  if (!mixedFixedRejected)
    return EXIT_FAILURE;

  auto const retiredOnlyConfiguration =
      configurationDirectory / "retired-only-v3.yaml";
  auto retiredOnlyYaml = std::ofstream{retiredOnlyConfiguration};
  retiredOnlyYaml << "schema_version: 3\n"
                     "tuning:\n"
                     "  mode: online_fixed\n"
                     "  strategy: random\n"
                     "  warmup_runs: 0\n"
                     "  runs_per_candidate: 1\n"
                     "  noise_cancellation_window: 1\n"
                     "  max_consecutive_runs: 1\n"
                     "  maximum_executions: null\n"
                     "  maximum_retired_configurations: 7\n"
                     "complete_history:\n"
                     "  file: retired-only-history.json\n";
  retiredOnlyYaml.close();
  auto const retiredOnlyDefaults =
      alpakaTune::TunerConfig::fromYaml(retiredOnlyConfiguration);
  if (retiredOnlyDefaults.mode != alpakaTune::TuningMode::onlineFixed ||
      retiredOnlyDefaults.maximumExecutions ||
      retiredOnlyDefaults.maximumRetiredConfigurations != 7u)
    return EXIT_FAILURE;

  auto sameFile = defaultConfig;
  sameFile.history.file = "shared.json";
  sameFile.completeHistory.file = "shared.json";
  auto sameFileRejected = false;
  try {
    sameFile.validate();
  } catch (std::invalid_argument const &) {
    sameFileRejected = true;
  }
  if (!sameFileRejected)
    return EXIT_FAILURE;

  auto zeroSampleCount = defaultConfig;
  zeroSampleCount.history.sampleCount = 0u;
  auto zeroSampleCountRejected = false;
  try {
    zeroSampleCount.validate();
  } catch (std::invalid_argument const &) {
    zeroSampleCountRejected = true;
  }
  if (!zeroSampleCountRejected)
    return EXIT_FAILURE;

  auto const legacyConfiguration = configurationDirectory / "legacy.yaml";
  auto legacyYaml = std::ofstream{legacyConfiguration};
  legacyYaml << "schema_version: 2\n"
                "tuning:\n"
                "  mode: online_adaptive\n"
                "  strategy: random\n"
                "  warmup_runs: 0\n"
                "  runs_per_candidate: 1\n"
                "  noise_cancellation_window: 1\n"
                "  max_consecutive_runs: 1\n"
                "  horizon: 100\n"
                "persistence:\n"
                "  file: legacy.json\n";
  legacyYaml.close();
  auto legacyRejected = false;
  try {
    static_cast<void>(alpakaTune::TunerConfig::fromYaml(legacyConfiguration));
  } catch (std::runtime_error const &) {
    legacyRejected = true;
  }
  if (!legacyRejected)
    return EXIT_FAILURE;

  std::filesystem::remove_all(configurationDirectory);
  return EXIT_SUCCESS;
}
