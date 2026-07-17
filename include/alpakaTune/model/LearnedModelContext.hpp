// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "alpakaTune/interfaces/Strategy.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace alpakaTune {

/** Device-specific output adapter selected by a learned model. */
enum class LearnedDeviceClass : std::uint32_t {
  cpu = 0u,
  gpu = 1u,
};

/** Semantic category of one tuning-space dimension. */
enum class LearnedDimensionKind : std::uint32_t {
  runtime = 0u,
  compileTime = 1u,
  launch = 2u,
  categorical = 3u,
};

/** A named, automatically derived scalar supplied to the learned model. */
struct LearnedContextFeature {
  std::string name;
  float value{};
};

/**
 * Description of one ordered tuning dimension.
 *
 * `concreteValues` is optional. When present it must contain one numeric value
 * for each discrete choice; categorical dimensions normally leave it empty.
 */
struct LearnedDimensionDescriptor {
  std::string name;
  LearnedDimensionKind kind{LearnedDimensionKind::runtime};
  std::size_t cardinality{};
  std::size_t componentIndex{};
  std::size_t vectorArity{1u};
  std::vector<float> concreteValues;
};

/**
 * Internal descriptor used to bridge Tuner-owned information to native model
 * inference without changing application-facing makeTuner/enqueue calls.
 */
struct LearnedModelContextDescriptor {
  static constexpr std::uint32_t featureSchemaVersion = 1u;

  LearnedDeviceClass deviceClass{LearnedDeviceClass::cpu};
  std::vector<LearnedContextFeature> contextFeatures;
  std::vector<LearnedDimensionDescriptor> dimensions;

  /**
   * Optional row-major legality mask in the same order as Tuner candidates.
   * An empty mask means that every point is legal.
   */
  std::vector<std::uint8_t> legalCandidates;
};

namespace detail {

inline constexpr std::size_t learnedDimensionFeatureCount = 18u;
inline constexpr std::size_t learnedDimensionNameHashBuckets = 8u;

inline constexpr std::string_view learnedArchitectureName =
    "deepsets_ensemble_v1";

inline constexpr std::string_view learnedDimensionFeatureNames[] = {
    "coordinate",         "concrete_value",     "cardinality_log1p",
    "dimension_position", "component_position", "arity_log1p",
    "kind_runtime",       "kind_compile_time",  "kind_launch",
    "kind_categorical",   "name_hash_0",        "name_hash_1",
    "name_hash_2",        "name_hash_3",        "name_hash_4",
    "name_hash_5",        "name_hash_6",        "name_hash_7",
};

[[nodiscard]] constexpr auto fnv1a64(std::string_view value) noexcept
    -> std::uint64_t {
  auto hash = std::uint64_t{14695981039346656037ull};
  for (auto const character : value) {
    hash ^= static_cast<std::uint8_t>(character);
    hash *= std::uint64_t{1099511628211ull};
  }
  return hash;
}

[[nodiscard]] inline auto signedHashFeatures(std::string_view value,
                                             std::size_t bucketCount)
    -> std::vector<float> {
  if (bucketCount == 0u)
    throw std::invalid_argument{
        "A signed hash feature vector needs at least one bucket."};
  auto features = std::vector<float>(bucketCount, 0.0f);
  auto const hash = fnv1a64(value);
  features.at(static_cast<std::size_t>(hash % bucketCount)) =
      (hash & (std::uint64_t{1u} << 63u)) == 0u ? 1.0f : -1.0f;
  return features;
}

[[nodiscard]] inline auto
candidateCount(std::span<LearnedDimensionDescriptor const> dimensions)
    -> std::size_t {
  auto count = std::size_t{1u};
  for (auto const &dimension : dimensions) {
    if (dimension.cardinality == 0u)
      throw std::invalid_argument{
          "Learned-model dimensions must not be empty."};
    if (count > std::numeric_limits<std::size_t>::max() / dimension.cardinality)
      throw std::overflow_error{
          "The learned-model candidate space is too large."};
    count *= dimension.cardinality;
  }
  return count;
}

[[nodiscard]] inline auto configurationForCandidate(
    std::size_t candidate,
    std::span<LearnedDimensionDescriptor const> dimensions)
    -> ParameterConfiguration {
  auto const count = candidateCount(dimensions);
  if (candidate >= count)
    throw std::out_of_range{
        "The learned-model candidate index is out of range."};
  auto configuration = ParameterConfiguration(dimensions.size(), 0.0f);
  for (auto position = dimensions.size(); position > 0u; --position) {
    auto const dimension = position - 1u;
    auto const cardinality = dimensions[dimension].cardinality;
    auto const index = candidate % cardinality;
    candidate /= cardinality;
    if (cardinality > 1u)
      configuration[dimension] =
          static_cast<float>(index) / static_cast<float>(cardinality - 1u);
  }
  return configuration;
}

[[nodiscard]] inline auto
validateLearnedContext(LearnedModelContextDescriptor const &context)
    -> std::string {
  if (context.dimensions.empty())
    return "A learned-model context needs at least one tuning dimension.";
  try {
    auto const count = candidateCount(context.dimensions);
    if (!context.legalCandidates.empty() &&
        context.legalCandidates.size() != count)
      return "The learned-model legality mask does not match the candidate "
             "space.";
  } catch (std::exception const &error) {
    return error.what();
  }

  auto featureNames = std::vector<std::string>{};
  featureNames.reserve(context.contextFeatures.size());
  for (auto const &feature : context.contextFeatures) {
    if (feature.name.empty())
      return "Learned-model context feature names must not be empty.";
    if (!std::isfinite(feature.value))
      return "Learned-model context features must be finite.";
    featureNames.push_back(feature.name);
  }
  std::ranges::sort(featureNames);
  if (std::ranges::adjacent_find(featureNames) != featureNames.end())
    return "Learned-model context feature names must be unique.";

  for (auto const &dimension : context.dimensions) {
    if (dimension.name.empty())
      return "Learned-model dimension names must not be empty.";
    if (static_cast<std::uint32_t>(dimension.kind) >
        static_cast<std::uint32_t>(LearnedDimensionKind::categorical))
      return "A learned-model dimension has an invalid kind.";
    if (dimension.vectorArity == 0u ||
        dimension.componentIndex >= dimension.vectorArity)
      return "A learned-model dimension has an invalid vector component.";
    if (!dimension.concreteValues.empty() &&
        dimension.concreteValues.size() != dimension.cardinality)
      return "A learned-model dimension has the wrong number of concrete "
             "values.";
    if (!std::ranges::all_of(dimension.concreteValues,
                             [](float value) { return std::isfinite(value); }))
      return "Learned-model concrete values must be finite.";
  }
  return {};
}

} // namespace detail

/** Stable signed FNV-1a feature hashing shared with offline tooling. */
[[nodiscard]] inline auto learnedSignedHashFeatures(std::string_view value,
                                                    std::size_t bucketCount)
    -> std::vector<float> {
  return detail::signedHashFeatures(value, bucketCount);
}

} // namespace alpakaTune
