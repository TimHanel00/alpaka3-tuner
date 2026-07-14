// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#include <alpakaTune/alpakaTune.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <utility>
#include <vector>

namespace {

class TestStrategyContext final : public alpakaTune::StrategyContext {
public:
  [[nodiscard]] auto parameterSizes() const noexcept
      -> std::span<std::size_t const> override {
    return m_sizes;
  }

  [[nodiscard]] auto runtimeFor(
      alpakaTune::ParameterConfiguration const &configuration) const
      -> std::optional<alpakaTune::RuntimeObservation> override {
    for (auto const &[knownConfiguration, runtime] : m_runtimes) {
      if (knownConfiguration == configuration)
        return alpakaTune::RuntimeObservation{
            runtime, 8u, 8u, alpakaTune::ConfigurationState::retired,
            alpakaTune::RuntimeComparison::inconclusive, true};
    }
    return std::nullopt;
  }

  void record(alpakaTune::ParameterConfiguration configuration, double runtime) {
    m_runtimes.emplace_back(std::move(configuration), runtime);
  }

private:
  std::array<std::size_t, 3u> m_sizes{2u, 3u, 4u};
  std::vector<std::pair<alpakaTune::ParameterConfiguration, double>> m_runtimes;
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
  for (auto const strategyKind : {alpakaTune::StrategyKind::random,
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

  if (alpakaTune::strategyFromName("exhaustive") != alpakaTune::StrategyKind::exhaustive ||
      alpakaTune::strategyFromName("random") != alpakaTune::StrategyKind::random ||
      alpakaTune::strategyFromName("simulated_annealing") !=
          alpakaTune::StrategyKind::simulatedAnnealing ||
      alpakaTune::strategyFromName("bayesian_optimization") !=
          alpakaTune::StrategyKind::bayesianOptimization)
    return EXIT_FAILURE;

  auto const configurationDirectory =
      std::filesystem::temp_directory_path() / "alpakaTune-strategy-yaml-test";
  std::filesystem::remove_all(configurationDirectory);
  std::filesystem::create_directories(configurationDirectory);
  for (auto const &[name, expected] :
       std::array<std::pair<char const *, alpakaTune::StrategyKind>, 4u>{
           {{"exhaustive", alpakaTune::StrategyKind::exhaustive},
            {"random", alpakaTune::StrategyKind::random},
            {"simulated_annealing", alpakaTune::StrategyKind::simulatedAnnealing},
            {"bayesian_optimization", alpakaTune::StrategyKind::bayesianOptimization}}}) {
    auto const configuration = configurationDirectory / (std::string{name} + ".yaml");
    auto yaml = std::ofstream{configuration};
    yaml << "schema_version: 1\n"
            "tuning:\n"
            "  strategy: "
         << name << "\n"
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
                    "persistence:\n"
                    "  directory: .alpakaTune\n";
    yaml.close();
    auto const &defaults = alpakaTune::session(configuration).defaults();
    if (defaults.strategy != expected || defaults.minimumRunsPerCandidate != 1u ||
        defaults.ciCheckInterval != 2u || std::abs(defaults.ciZScore - 1.96) > 1.0e-12 ||
        std::abs(defaults.ciRelativeWidth - 0.1) > 1.0e-12 ||
        std::abs(defaults.outlierMadScale - 4.0) > 1.0e-12 ||
        defaults.mannWhitneyEarlyStop || defaults.mannWhitneyMinimumSamples != 8u ||
        std::abs(defaults.mannWhitneyAlpha - 0.1) > 1.0e-12)
      return EXIT_FAILURE;
  }
  std::filesystem::remove_all(configurationDirectory);
  return EXIT_SUCCESS;
}
