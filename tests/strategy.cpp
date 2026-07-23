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
      defaultConfig.maximumExecutions != 40'000u ||
      defaultConfig.maximumRetiredConfigurations ||
      defaultConfig.persistenceFile || !defaultConfig.persistenceRead ||
      !defaultConfig.persistenceWrite)
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
    yaml << "schema_version: 1\n"
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
            "persistence:\n"
            "  file: .alpakaTune/history.json\n";
    yaml.close();
    auto const defaults = alpakaTune::TunerConfig::fromYaml(configuration);
    if (defaults.strategy != expected ||
        defaults.persistenceFile != ".alpakaTune/history.json" ||
        !defaults.persistenceRead || !defaults.persistenceWrite ||
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
                "  max_consecutive_runs: 1\n";
  memoryYaml.close();
  auto const memoryDefaults =
      alpakaTune::TunerConfig::fromYaml(memoryConfiguration);
  if (memoryDefaults.persistenceFile || !memoryDefaults.persistenceRead ||
      !memoryDefaults.persistenceWrite)
    return EXIT_FAILURE;

  auto const processLocalConfiguration =
      configurationDirectory / "process-local.yaml";
  auto processLocalYaml = std::ofstream{processLocalConfiguration};
  processLocalYaml << "schema_version: 2\n"
                      "tuning:\n"
                      "  mode: online_adaptive\n"
                      "  strategy: random\n"
                      "  warmup_runs: 0\n"
                      "  runs_per_candidate: 1\n"
                      "  noise_cancellation_window: 1\n"
                      "  max_consecutive_runs: 1\n"
                      "persistence:\n"
                      "  read: false\n"
                      "  write: false\n";
  processLocalYaml.close();
  auto const processLocalDefaults =
      alpakaTune::TunerConfig::fromYaml(processLocalConfiguration);
  if (processLocalDefaults.persistenceFile ||
      processLocalDefaults.persistenceRead ||
      processLocalDefaults.persistenceWrite)
    return EXIT_FAILURE;

  auto const accessConfiguration =
      configurationDirectory / "history-access.yaml";
  auto accessYaml = std::ofstream{accessConfiguration};
  accessYaml << "schema_version: 2\n"
                "tuning:\n"
                "  mode: offline\n"
                "  strategy: random\n"
                "  warmup_runs: 0\n"
                "  runs_per_candidate: 1\n"
                "  noise_cancellation_window: 1\n"
                "  max_consecutive_runs: 1\n"
                "persistence:\n"
                "  file: read-only-history.json\n"
                "  read: true\n"
                "  write: false\n";
  accessYaml.close();
  auto const accessDefaults =
      alpakaTune::TunerConfig::fromYaml(accessConfiguration);
  if (accessDefaults.persistenceFile != "read-only-history.json" ||
      !accessDefaults.persistenceRead || accessDefaults.persistenceWrite)
    return EXIT_FAILURE;

  auto const writeOnlyConfiguration =
      configurationDirectory / "write-only-history.yaml";
  auto writeOnlyYaml = std::ofstream{writeOnlyConfiguration};
  writeOnlyYaml << "schema_version: 2\n"
                   "tuning:\n"
                   "  mode: online_fixed\n"
                   "  strategy: random\n"
                   "  warmup_runs: 0\n"
                   "  runs_per_candidate: 1\n"
                   "  noise_cancellation_window: 1\n"
                   "  max_consecutive_runs: 1\n"
                   "persistence:\n"
                   "  file: fresh-history.json\n"
                   "  read: false\n"
                   "  write: true\n";
  writeOnlyYaml.close();
  auto const writeOnlyDefaults =
      alpakaTune::TunerConfig::fromYaml(writeOnlyConfiguration);
  if (writeOnlyDefaults.persistenceFile != "fresh-history.json" ||
      writeOnlyDefaults.persistenceRead || !writeOnlyDefaults.persistenceWrite)
    return EXIT_FAILURE;

  auto const learnedConfiguration = configurationDirectory / "learned-v2.yaml";
  auto learnedYaml = std::ofstream{learnedConfiguration};
  learnedYaml << "schema_version: 2\n"
                 "tuning:\n"
                 "  mode: online_fixed\n"
                 "  strategy: learned_hybrid\n"
                 "  warmup_runs: 0\n"
                 "  runs_per_candidate: 1\n"
                 "  noise_cancellation_window: 1\n"
                 "  max_consecutive_runs: 1\n"
                 "  maximum_executions: 100000\n"
                 "persistence:\n"
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
      configurationDirectory / "adaptive-v2.yaml";
  auto adaptiveYaml = std::ofstream{adaptiveConfiguration};
  adaptiveYaml << "schema_version: 2\n"
                  "tuning:\n"
                  "  mode: online_adaptive\n"
                  "  strategy: random\n"
                  "  warmup_runs: 1\n"
                  "  runs_per_candidate: 1\n"
                  "  noise_cancellation_window: 2\n"
                  "  max_consecutive_runs: 4\n"
                  "  maximum_executions: 4000\n"
                  "  maximum_retired_configurations: 1\n"
                  "  history_window_size: 10\n"
                  "  revisit_admission_steepness: 16\n"
                  "  score_temperature_start: 0.25\n"
                  "  score_temperature_end: 0.05\n"
                  "persistence:\n"
                  "  file: adaptive-history.json\n";
  adaptiveYaml.close();
  auto const adaptiveDefaults =
      alpakaTune::TunerConfig::fromYaml(adaptiveConfiguration);
  if (adaptiveDefaults.mode != alpakaTune::TuningMode::onlineAdaptive ||
      adaptiveDefaults.maximumExecutions != 4000u ||
      adaptiveDefaults.maximumRetiredConfigurations != 1u ||
      adaptiveDefaults.historyWindowSize != 10u ||
      std::abs(adaptiveDefaults.revisitAdmissionSteepness - 16.0) > 1.0e-12 ||
      std::abs(adaptiveDefaults.scoreTemperatureStart - 0.25) > 1.0e-12 ||
      std::abs(adaptiveDefaults.scoreTemperatureEnd - 0.05) > 1.0e-12)
    return EXIT_FAILURE;

  auto const retiredOnlyConfiguration =
      configurationDirectory / "retired-only-v2.yaml";
  auto retiredOnlyYaml = std::ofstream{retiredOnlyConfiguration};
  retiredOnlyYaml << "schema_version: 2\n"
                     "tuning:\n"
                     "  mode: online_fixed\n"
                     "  strategy: random\n"
                     "  warmup_runs: 0\n"
                     "  runs_per_candidate: 1\n"
                     "  noise_cancellation_window: 1\n"
                     "  max_consecutive_runs: 1\n"
                     "  maximum_executions: null\n"
                     "  maximum_retired_configurations: 7\n"
                     "persistence:\n"
                     "  file: retired-only-history.json\n";
  retiredOnlyYaml.close();
  auto const retiredOnlyDefaults =
      alpakaTune::TunerConfig::fromYaml(retiredOnlyConfiguration);
  if (retiredOnlyDefaults.mode != alpakaTune::TuningMode::onlineFixed ||
      retiredOnlyDefaults.maximumExecutions ||
      retiredOnlyDefaults.maximumRetiredConfigurations != 7u)
    return EXIT_FAILURE;

  std::filesystem::remove_all(configurationDirectory);
  return EXIT_SUCCESS;
}
