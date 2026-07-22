// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#include <alpakaTune/strategy/LearnedHybridStrategy.hpp>

#include <nlohmann/json.hpp>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

struct TensorWriter {
  void add(std::string name, std::vector<std::size_t> shape,
           std::vector<float> tensorValues) {
    tensors.push_back({{"name", std::move(name)},
                       {"dtype", "float32"},
                       {"shape", std::move(shape)}});
    values.insert(values.end(), tensorValues.begin(), tensorValues.end());
  }

  nlohmann::json tensors = nlohmann::json::array();
  std::vector<float> values;
};

auto zeros(std::size_t count) -> std::vector<float> {
  return std::vector<float>(count, 0.0f);
}

void writeU32(std::ofstream &output, std::uint32_t value) {
  auto bytes = std::array<char, 4u>{static_cast<char>(value & 0xffu),
                                    static_cast<char>((value >> 8u) & 0xffu),
                                    static_cast<char>((value >> 16u) & 0xffu),
                                    static_cast<char>((value >> 24u) & 0xffu)};
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void writeFloat(std::ofstream &output, float value) {
  writeU32(output, std::bit_cast<std::uint32_t>(value));
}

auto makeArtifact(std::filesystem::path const &path) -> void {
  constexpr std::size_t dimensions = 18u;
  constexpr std::size_t token0Width = 2u;
  constexpr std::size_t token2Width = 3u;
  constexpr std::size_t context = 1u;
  constexpr std::size_t embedding = 2u;
  constexpr std::size_t members = 3u;
  auto writer = TensorWriter{};
  for (std::size_t member = 0u; member < members; ++member) {
    auto const prefix = std::string{"members."} + std::to_string(member) + '.';
    auto token0 = zeros(token0Width * dimensions);
    token0[0] = 1.0f; // coordinate -> first hidden unit
    writer.add(prefix + "token.0.weight", {token0Width, dimensions}, token0);
    writer.add(prefix + "token.0.bias", {token0Width}, zeros(token0Width));
    auto token2 = zeros(token2Width * token0Width);
    token2[0] = 1.0f;
    writer.add(prefix + "token.2.weight", {token2Width, token0Width}, token2);
    writer.add(prefix + "token.2.bias", {token2Width}, zeros(token2Width));

    auto context0 = zeros(embedding * (2u * token2Width + context));
    context0[0] = 1.0f; // mean-pooled first token unit
    writer.add(prefix + "context.0.weight",
               {embedding, 2u * token2Width + context}, context0);
    writer.add(prefix + "context.0.bias", {embedding}, zeros(embedding));
    auto context2 = zeros(embedding * embedding);
    context2[0] = 1.0f;
    writer.add(prefix + "context.2.weight", {embedding, embedding}, context2);
    writer.add(prefix + "context.2.bias", {embedding}, zeros(embedding));
    writer.add(prefix + "adapters.cpu.weight", {1u, embedding}, {1.0f, 0.0f});
    writer.add(prefix + "adapters.cpu.bias", {1u},
               {static_cast<float>(member) * 0.1f});
    writer.add(prefix + "adapters.gpu.weight", {1u, embedding}, {1.0f, 0.0f});
    writer.add(prefix + "adapters.gpu.bias", {1u},
               {static_cast<float>(member) * 0.1f});
  }

  auto dimensionNames = nlohmann::json::array();
  for (auto const name : alpakaTune::detail::learnedDimensionFeatureNames)
    dimensionNames.push_back(name);
  auto metadata = nlohmann::json{
      {"artifact_version", 1},
      {"feature_schema_version", 1},
      {"architecture", "deepsets_ensemble_v1"},
      {"ensemble_size", members},
      {"context_feature_count", context},
      {"dimension_feature_count", dimensions},
      {"token_hidden_sizes", {token0Width, token2Width}},
      {"embedding_size", embedding},
      {"device_class_count", 2},
      {"adapter_feature_count", 2},
      {"feature_names",
       {{"context", {"candidate_count_log1p"}},
        {"dimension", std::move(dimensionNames)}}},
      {"hash_buckets", {{"dimension_name", 8}}},
      {"preprocessing",
       {{"context_mean", {0.0f}},
        {"context_scale", {1.0f}},
        {"dimension_mean", zeros(dimensions)},
        {"dimension_scale", std::vector<float>(dimensions, 1.0f)}}},
      {"tensors", writer.tensors}};
  auto const encoded = metadata.dump();
  auto output = std::ofstream{path, std::ios::binary};
  output.write(alpakaTune::LearnedModelArtifact::magic.data(),
               static_cast<std::streamsize>(
                   alpakaTune::LearnedModelArtifact::magic.size()));
  writeU32(output, static_cast<std::uint32_t>(encoded.size()));
  output.write(encoded.data(), static_cast<std::streamsize>(encoded.size()));
  for (auto const value : writer.values)
    writeFloat(output, value);
}

class Context final : public alpakaTune::StrategyContext {
public:
  [[nodiscard]] auto parameterSizes() const noexcept
      -> std::span<std::size_t const> override {
    return sizes;
  }

  [[nodiscard]] auto
  runtimeFor(alpakaTune::ParameterConfiguration const &configuration) const
      -> std::optional<alpakaTune::RuntimeObservation> override {
    for (auto const &[known, seconds] : observations)
      if (known == configuration)
        return alpakaTune::RuntimeObservation{
            seconds,
            3u,
            3u,
            alpakaTune::ConfigurationState::retired,
            alpakaTune::RuntimeComparison::inconclusive,
            true};
    return std::nullopt;
  }

  void record(alpakaTune::ParameterConfiguration configuration,
              double seconds) {
    observations.emplace_back(std::move(configuration), seconds);
  }

  std::vector<std::size_t> sizes{6u};
  std::vector<std::pair<alpakaTune::ParameterConfiguration, double>>
      observations;
};

auto descriptor(std::size_t cardinality = 6u)
    -> alpakaTune::LearnedModelContextDescriptor {
  auto concreteValues = std::vector<float>(cardinality);
  for (std::size_t index = 0u; index < cardinality; ++index)
    concreteValues[index] = static_cast<float>(index + 1u);
  return {
      .deviceClass = alpakaTune::LearnedDeviceClass::gpu,
      .contextFeatures = {
          {"candidate_count_log1p", std::log1p(static_cast<float>(cardinality))}},
      .dimensions = {{.name = "block_size",
                      .kind = alpakaTune::LearnedDimensionKind::launch,
                      .cardinality = cardinality,
                      .componentIndex = 0u,
                      .vectorArity = 1u,
                      .concreteValues = std::move(concreteValues)}},
      .legalCandidates = {},
  };
}

} // namespace

auto main() -> int {
  auto const directory = std::filesystem::temp_directory_path() /
                         "alpakaTune-learned-strategy-test";
  std::filesystem::create_directories(directory);
  auto const modelPath = directory / "tiny.atml";
  makeArtifact(modelPath);

  auto const loaded = alpakaTune::loadLearnedModelArtifact(modelPath);
  if (!loaded || loaded.artifact->ensembleSize() != 3u ||
      loaded.artifact->dimensionFeatureCount() != 18u)
    return EXIT_FAILURE;

  auto model = alpakaTune::NativeDeepSetsModel{loaded.artifact, descriptor()};
  auto const slow = model.predict({1.0f});
  auto const fast = model.predict({0.0f});
  auto const batch = model.predictBatch(
      std::array<alpakaTune::ParameterConfiguration, 2u>{{{1.0f}, {0.0f}}});
  if (!(fast.logRuntimeSeconds < slow.logRuntimeSeconds) ||
      !(fast.uncertainty > 0.0) || batch.size() != 2u ||
      batch[0].logRuntimeSeconds != slow.logRuntimeSeconds ||
      batch[1].logRuntimeSeconds != fast.logRuntimeSeconds) {
    std::cerr << "scalar/batch prediction mismatch\n";
    return EXIT_FAILURE;
  }

  auto options = alpakaTune::LearnedHybridOptions{};
  options.adapterBatchSize = 2u;
  auto strategy = alpakaTune::LearnedHybridStrategy{loaded.artifact,
                                                    descriptor(), 17u, options};
  auto context = Context{};
  auto const first = strategy.recommend(context);
  context.record(first, std::exp(0.5));
  auto const second = strategy.recommend(context);
  context.record(second, std::exp(0.6));
  auto const third = strategy.recommend(context);
  auto const fourth = strategy.recommend(context);
  auto const fifth = strategy.recommend(context);
  if (first != alpakaTune::ParameterConfiguration{0.0f} ||
      strategy.status() != alpakaTune::LearnedHybridStatus::active ||
      strategy.adapterUpdateCount() != 1u ||
      strategy.incorporatedObservationCount() != 2u ||
      strategy.cachedCandidateCount() > 6u ||
      strategy.peakCachedCandidateCount() != 6u || fifth.front() <= 0.6f ||
      strategy.lastSelectionReason() !=
          alpakaTune::LearnedSelectionReason::uncertaintyDiversity ||
      second == third || third == fourth) {
    std::cerr << "small-space learned behavior mismatch: first=" << first.front()
              << " updates=" << strategy.adapterUpdateCount()
              << " observations=" << strategy.incorporatedObservationCount()
              << " cached=" << strategy.cachedCandidateCount()
              << " fifth=" << fifth.front()
              << " reason=" << static_cast<int>(strategy.lastSelectionReason())
              << " second=" << second.front() << " third=" << third.front()
              << " fourth=" << fourth.front() << '\n';
    return EXIT_FAILURE;
  }

  auto boundedOptions = options;
  boundedOptions.candidatePoolSize = 4u;
  boundedOptions.candidateBatchSize = 2u;
  auto boundedContext = Context{};
  boundedContext.sizes = {24u};
  auto bounded = alpakaTune::LearnedHybridStrategy{
      loaded.artifact, descriptor(24u), 31u, boundedOptions};
  auto deterministic = alpakaTune::LearnedHybridStrategy{
      loaded.artifact, descriptor(24u), 31u, boundedOptions};
  auto seen = std::vector<alpakaTune::ParameterConfiguration>{};
  for (std::size_t index = 0u; index < 24u; ++index) {
    auto const candidate = bounded.recommend(boundedContext);
    auto const repeated = std::ranges::find(seen, candidate) != seen.end();
    if (repeated || candidate != deterministic.recommend(boundedContext)) {
      std::cerr << "bounded sequence mismatch at " << index << '\n';
      return EXIT_FAILURE;
    }
    seen.push_back(candidate);
  }
  auto const exhaustedRepeat = bounded.recommend(boundedContext);
  if (std::ranges::find(seen, exhaustedRepeat) == seen.end() ||
      bounded.cachedCandidateCount() > 4u ||
      bounded.peakCachedCandidateCount() > 4u ||
      bounded.scoredCandidateCount() != 24u ||
      bounded.poolRefillCount() <= 1u ||
      !bounded.candidateStreamExhausted()) {
    std::cerr << "bounded diagnostics mismatch: cached="
              << bounded.cachedCandidateCount()
              << " peak=" << bounded.peakCachedCandidateCount()
              << " scored=" << bounded.scoredCandidateCount()
              << " refills=" << bounded.poolRefillCount()
              << " exhausted=" << bounded.candidateStreamExhausted() << '\n';
    return EXIT_FAILURE;
  }

  auto restrictedDescriptor = descriptor();
  restrictedDescriptor.legalCandidates = {1u, 0u, 0u, 0u, 0u, 0u};
  auto restricted = alpakaTune::LearnedHybridStrategy{
      loaded.artifact, std::move(restrictedDescriptor), 23u, options};
  auto const onlyLegalCandidate = restricted.recommend(context);
  auto const repeatedCandidate = restricted.recommend(context);
  if (onlyLegalCandidate != alpakaTune::ParameterConfiguration{0.0f} ||
      repeatedCandidate != onlyLegalCandidate ||
      restricted.status() != alpakaTune::LearnedHybridStatus::active)
    return EXIT_FAILURE;

  auto missing = alpakaTune::LearnedHybridStrategy{directory / "missing.atml",
                                                   descriptor(), 9u};
  auto const fallback = missing.recommend(context);
  if (missing.status() !=
          alpakaTune::LearnedHybridStatus::fallbackArtifactUnavailable ||
      missing.artifactLoadStatus() !=
          alpakaTune::LearnedModelLoadStatus::fileNotFound ||
      fallback.size() != 1u)
    return EXIT_FAILURE;

  std::filesystem::remove(modelPath);
  std::filesystem::remove(directory);
  return EXIT_SUCCESS;
}
