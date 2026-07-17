// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "alpakaTune/model/LearnedModelContext.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(ALPAKA_TUNE_HAS_JSON) && ALPAKA_TUNE_HAS_JSON
#include <nlohmann/json.hpp>
#endif

namespace alpakaTune {

namespace detail {
struct LearnedModelArtifactBuilder;
}

/** Result category for loading the stable ATMLART1 deployment artifact. */
enum class LearnedModelLoadStatus {
  available,
  jsonUnavailable,
  fileNotFound,
  ioError,
  invalidMagic,
  invalidMetadata,
  unsupportedVersion,
  incompatibleFeatureSchema,
  invalidPayload,
};

[[nodiscard]] constexpr auto
learnedModelLoadStatusName(LearnedModelLoadStatus status) noexcept
    -> std::string_view {
  switch (status) {
  case LearnedModelLoadStatus::available:
    return "available";
  case LearnedModelLoadStatus::jsonUnavailable:
    return "json_unavailable";
  case LearnedModelLoadStatus::fileNotFound:
    return "file_not_found";
  case LearnedModelLoadStatus::ioError:
    return "io_error";
  case LearnedModelLoadStatus::invalidMagic:
    return "invalid_magic";
  case LearnedModelLoadStatus::invalidMetadata:
    return "invalid_metadata";
  case LearnedModelLoadStatus::unsupportedVersion:
    return "unsupported_version";
  case LearnedModelLoadStatus::incompatibleFeatureSchema:
    return "incompatible_feature_schema";
  case LearnedModelLoadStatus::invalidPayload:
    return "invalid_payload";
  }
  return "unknown";
}

struct LearnedModelTensor {
  std::string name;
  std::vector<std::size_t> shape;
  std::size_t offset{};
  std::size_t elementCount{};
};

/** Parsed, immutable model artifact independent of the training framework. */
class LearnedModelArtifact {
public:
  static constexpr std::array<char, 8u> magic{'A', 'T', 'M', 'L',
                                              'A', 'R', 'T', '1'};
  static constexpr std::uint32_t artifactVersion = 1u;

  [[nodiscard]] auto tensor(std::string_view name) const
      -> std::optional<std::span<float const>> {
    auto const found =
        std::ranges::find(m_tensors, name, &LearnedModelTensor::name);
    if (found == m_tensors.end())
      return std::nullopt;
    return std::span<float const>{m_values}.subspan(found->offset,
                                                    found->elementCount);
  }

  [[nodiscard]] auto tensorDescriptor(std::string_view name) const
      -> LearnedModelTensor const * {
    auto const found =
        std::ranges::find(m_tensors, name, &LearnedModelTensor::name);
    return found == m_tensors.end() ? nullptr : std::addressof(*found);
  }

  [[nodiscard]] auto architecture() const noexcept -> std::string_view {
    return m_architecture;
  }
  [[nodiscard]] auto featureSchema() const noexcept -> std::uint32_t {
    return m_featureSchemaVersion;
  }
  [[nodiscard]] auto ensembleSize() const noexcept -> std::size_t {
    return m_ensembleSize;
  }
  [[nodiscard]] auto contextFeatureCount() const noexcept -> std::size_t {
    return m_contextFeatureCount;
  }
  [[nodiscard]] auto dimensionFeatureCount() const noexcept -> std::size_t {
    return m_dimensionFeatureCount;
  }
  [[nodiscard]] auto tokenHiddenSizes() const noexcept
      -> std::span<std::size_t const> {
    return m_tokenHiddenSizes;
  }
  [[nodiscard]] auto embeddingSize() const noexcept -> std::size_t {
    return m_embeddingSize;
  }
  [[nodiscard]] auto deviceClassCount() const noexcept -> std::size_t {
    return m_deviceClassCount;
  }
  [[nodiscard]] auto adapterFeatureCount() const noexcept -> std::size_t {
    return m_adapterFeatureCount;
  }
  [[nodiscard]] auto contextFeatureNames() const noexcept
      -> std::span<std::string const> {
    return m_contextFeatureNames;
  }
  [[nodiscard]] auto dimensionFeatureNames() const noexcept
      -> std::span<std::string const> {
    return m_dimensionFeatureNames;
  }
  [[nodiscard]] auto contextMean() const noexcept -> std::span<float const> {
    return m_contextMean;
  }
  [[nodiscard]] auto contextScale() const noexcept -> std::span<float const> {
    return m_contextScale;
  }
  [[nodiscard]] auto dimensionMean() const noexcept -> std::span<float const> {
    return m_dimensionMean;
  }
  [[nodiscard]] auto dimensionScale() const noexcept -> std::span<float const> {
    return m_dimensionScale;
  }
  [[nodiscard]] auto metadataJson() const noexcept -> std::string_view {
    return m_metadataJson;
  }
  [[nodiscard]] auto uncertaintyHead() const noexcept -> std::string_view {
    return m_uncertaintyHead;
  }

private:
  friend struct detail::LearnedModelArtifactBuilder;

  std::string m_metadataJson;
  std::string m_architecture;
  std::uint32_t m_featureSchemaVersion{};
  std::size_t m_ensembleSize{};
  std::size_t m_contextFeatureCount{};
  std::size_t m_dimensionFeatureCount{};
  std::vector<std::size_t> m_tokenHiddenSizes;
  std::size_t m_embeddingSize{};
  std::size_t m_deviceClassCount{};
  std::size_t m_adapterFeatureCount{};
  std::string m_uncertaintyHead;
  std::vector<std::string> m_contextFeatureNames;
  std::vector<std::string> m_dimensionFeatureNames;
  std::vector<float> m_contextMean;
  std::vector<float> m_contextScale;
  std::vector<float> m_dimensionMean;
  std::vector<float> m_dimensionScale;
  std::vector<LearnedModelTensor> m_tensors;
  std::vector<float> m_values;
};

struct LearnedModelLoadResult {
  LearnedModelLoadStatus status{LearnedModelLoadStatus::invalidMetadata};
  std::string message;
  std::shared_ptr<LearnedModelArtifact const> artifact;

  [[nodiscard]] explicit operator bool() const noexcept {
    return status == LearnedModelLoadStatus::available && artifact != nullptr;
  }
};

namespace detail {

struct LearnedModelArtifactBuilder {
  static constexpr std::size_t maximumMetadataBytes = 16u * 1024u * 1024u;
  static constexpr std::size_t maximumTensorElements =
      std::size_t{512u} * 1024u * 1024u / sizeof(float);

#if defined(ALPAKA_TUNE_HAS_JSON) && ALPAKA_TUNE_HAS_JSON
  [[nodiscard]] static auto sizeField(nlohmann::json const &metadata,
                                      char const *name) -> std::size_t {
    auto const value = metadata.at(name).get<std::uint64_t>();
    if (value == 0u || value > std::numeric_limits<std::size_t>::max())
      throw std::invalid_argument{
          std::string{"Invalid positive metadata field: "} + name};
    return static_cast<std::size_t>(value);
  }

  [[nodiscard]] static auto stringArray(nlohmann::json const &value,
                                        std::size_t expected, char const *name)
      -> std::vector<std::string> {
    if (!value.is_array() || value.size() != expected)
      throw std::invalid_argument{std::string{"Invalid feature-name array: "} +
                                  name};
    auto result = std::vector<std::string>{};
    result.reserve(expected);
    auto unique = std::unordered_set<std::string>{};
    for (auto const &entry : value) {
      auto item = entry.get<std::string>();
      if (item.empty() || !unique.insert(item).second)
        throw std::invalid_argument{
            std::string{"Feature names must be non-empty and unique: "} + name};
      result.push_back(std::move(item));
    }
    return result;
  }

  [[nodiscard]] static auto positiveSizeArray(nlohmann::json const &value,
                                              std::size_t expected,
                                              char const *name)
      -> std::vector<std::size_t> {
    if (!value.is_array() || value.size() != expected)
      throw std::invalid_argument{std::string{"Invalid size array: "} + name};
    auto result = std::vector<std::size_t>{};
    result.reserve(expected);
    for (auto const &entry : value) {
      auto const item = entry.get<std::uint64_t>();
      if (item == 0u || item > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument{std::string{"Invalid positive size: "} +
                                    name};
      result.push_back(static_cast<std::size_t>(item));
    }
    return result;
  }

  [[nodiscard]] static auto floatArray(nlohmann::json const &value,
                                       std::size_t expected, char const *name,
                                       bool positive = false)
      -> std::vector<float> {
    if (!value.is_array() || value.size() != expected)
      throw std::invalid_argument{std::string{"Invalid preprocessing array: "} +
                                  name};
    auto result = std::vector<float>{};
    result.reserve(expected);
    for (auto const &entry : value) {
      auto const number = entry.get<float>();
      if (!std::isfinite(number) || (positive && number <= 0.0f))
        throw std::invalid_argument{
            std::string{"Invalid preprocessing value: "} + name};
      result.push_back(number);
    }
    return result;
  }

  [[nodiscard]] static auto
  build(std::string metadataText, nlohmann::json const &metadata,
        std::vector<float> values, std::vector<LearnedModelTensor> tensors)
      -> std::shared_ptr<LearnedModelArtifact const> {
    auto artifact = std::make_shared<LearnedModelArtifact>();
    artifact->m_metadataJson = std::move(metadataText);
    artifact->m_architecture = metadata.at("architecture").get<std::string>();
    artifact->m_featureSchemaVersion =
        metadata.at("feature_schema_version").get<std::uint32_t>();
    artifact->m_ensembleSize = sizeField(metadata, "ensemble_size");
    artifact->m_contextFeatureCount =
        sizeField(metadata, "context_feature_count");
    artifact->m_dimensionFeatureCount =
        sizeField(metadata, "dimension_feature_count");
    if (metadata.contains("token_hidden_sizes"))
      artifact->m_tokenHiddenSizes = positiveSizeArray(
          metadata.at("token_hidden_sizes"), 2u, "token_hidden_sizes");
    else {
      auto const legacy = sizeField(metadata, "token_hidden_size");
      artifact->m_tokenHiddenSizes = {legacy, legacy};
    }
    artifact->m_embeddingSize = sizeField(metadata, "embedding_size");
    artifact->m_deviceClassCount = sizeField(metadata, "device_class_count");
    artifact->m_adapterFeatureCount =
        metadata.contains("adapter_feature_count")
            ? sizeField(metadata, "adapter_feature_count")
            : std::min<std::size_t>(8u, artifact->m_embeddingSize);
    artifact->m_uncertaintyHead =
        metadata.value("uncertainty_head", std::string{});
    if (!artifact->m_uncertaintyHead.empty() &&
        artifact->m_uncertaintyHead != "softplus_stddev")
      throw std::invalid_argument{
          "The learned-model uncertainty head is unsupported."};

    auto const &features = metadata.at("feature_names");
    artifact->m_contextFeatureNames =
        stringArray(features.at("context"), artifact->m_contextFeatureCount,
                    "feature_names.context");
    artifact->m_dimensionFeatureNames =
        stringArray(features.at("dimension"), artifact->m_dimensionFeatureCount,
                    "feature_names.dimension");

    auto const &hashBuckets = metadata.at("hash_buckets");
    if (hashBuckets.at("dimension_name").get<std::size_t>() !=
        learnedDimensionNameHashBuckets)
      throw std::invalid_argument{
          "Unsupported dimension-name hash bucket count."};

    auto const &preprocessing = metadata.at("preprocessing");
    artifact->m_contextMean =
        floatArray(preprocessing.at("context_mean"),
                   artifact->m_contextFeatureCount, "context_mean");
    artifact->m_contextScale =
        floatArray(preprocessing.at("context_scale"),
                   artifact->m_contextFeatureCount, "context_scale", true);
    artifact->m_dimensionMean =
        floatArray(preprocessing.at("dimension_mean"),
                   artifact->m_dimensionFeatureCount, "dimension_mean");
    artifact->m_dimensionScale =
        floatArray(preprocessing.at("dimension_scale"),
                   artifact->m_dimensionFeatureCount, "dimension_scale", true);

    artifact->m_tensors = std::move(tensors);
    artifact->m_values = std::move(values);
    return artifact;
  }
#endif
};

[[nodiscard]] inline auto littleEndianU32(std::array<char, 4u> const &bytes)
    -> std::uint32_t {
  return static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[0])) |
         (static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[1]))
          << 8u) |
         (static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[2]))
          << 16u) |
         (static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[3]))
          << 24u);
}

[[nodiscard]] inline auto invalidModelLoad(LearnedModelLoadStatus status,
                                           std::string message)
    -> LearnedModelLoadResult {
  return {.status = status, .message = std::move(message), .artifact = nullptr};
}

} // namespace detail

/**
 * Load an ATMLART1 file: magic, little-endian JSON byte count, canonical UTF-8
 * metadata, then contiguous little-endian float32 tensors in metadata order.
 */
[[nodiscard]] inline auto
loadLearnedModelArtifact(std::filesystem::path const &path)
    -> LearnedModelLoadResult {
#if !defined(ALPAKA_TUNE_HAS_JSON) || !ALPAKA_TUNE_HAS_JSON
  static_cast<void>(path);
  return detail::invalidModelLoad(
      LearnedModelLoadStatus::jsonUnavailable,
      "Learned-model artifacts require alpakaTune built with JSON support.");
#else
  if (!std::filesystem::exists(path))
    return detail::invalidModelLoad(
        LearnedModelLoadStatus::fileNotFound,
        "The learned-model artifact does not exist.");
  auto input = std::ifstream{path, std::ios::binary};
  if (!input)
    return detail::invalidModelLoad(
        LearnedModelLoadStatus::ioError,
        "The learned-model artifact cannot be opened.");

  auto magic = std::array<char, 8u>{};
  auto metadataLengthBytes = std::array<char, 4u>{};
  if (!input.read(magic.data(), static_cast<std::streamsize>(magic.size())) ||
      !input.read(metadataLengthBytes.data(),
                  static_cast<std::streamsize>(metadataLengthBytes.size())))
    return detail::invalidModelLoad(
        LearnedModelLoadStatus::ioError,
        "The learned-model artifact header is truncated.");
  if (magic != LearnedModelArtifact::magic)
    return detail::invalidModelLoad(
        LearnedModelLoadStatus::invalidMagic,
        "The learned-model artifact magic is invalid.");
  auto const metadataLength = detail::littleEndianU32(metadataLengthBytes);
  if (metadataLength == 0u ||
      metadataLength >
          detail::LearnedModelArtifactBuilder::maximumMetadataBytes)
    return detail::invalidModelLoad(
        LearnedModelLoadStatus::invalidMetadata,
        "The learned-model metadata length is invalid.");
  auto metadataText = std::string(metadataLength, '\0');
  if (!input.read(metadataText.data(),
                  static_cast<std::streamsize>(metadataLength)))
    return detail::invalidModelLoad(LearnedModelLoadStatus::ioError,
                                    "The learned-model metadata is truncated.");

  auto const metadata = nlohmann::json::parse(metadataText, nullptr, false);
  if (metadata.is_discarded() || !metadata.is_object())
    return detail::invalidModelLoad(
        LearnedModelLoadStatus::invalidMetadata,
        "The learned-model metadata is not valid JSON.");
  try {
    if (metadata.at("artifact_version").get<std::uint32_t>() !=
        LearnedModelArtifact::artifactVersion)
      return detail::invalidModelLoad(
          LearnedModelLoadStatus::unsupportedVersion,
          "The learned-model artifact version is unsupported.");
    if (metadata.at("feature_schema_version").get<std::uint32_t>() !=
        LearnedModelContextDescriptor::featureSchemaVersion)
      return detail::invalidModelLoad(
          LearnedModelLoadStatus::incompatibleFeatureSchema,
          "The learned-model feature schema is incompatible with this build.");
    if (metadata.at("architecture").get<std::string>() !=
        detail::learnedArchitectureName)
      return detail::invalidModelLoad(
          LearnedModelLoadStatus::invalidMetadata,
          "The learned-model architecture is unsupported.");

    auto tensors = std::vector<LearnedModelTensor>{};
    auto names = std::unordered_set<std::string>{};
    auto totalElements = std::size_t{};
    auto const &tensorMetadata = metadata.at("tensors");
    if (!tensorMetadata.is_array() || tensorMetadata.empty())
      throw std::invalid_argument{
          "The artifact tensor table must not be empty."};
    tensors.reserve(tensorMetadata.size());
    for (auto const &entry : tensorMetadata) {
      auto tensor = LearnedModelTensor{};
      tensor.name = entry.at("name").get<std::string>();
      if (tensor.name.empty() || !names.insert(tensor.name).second)
        throw std::invalid_argument{
            "Artifact tensor names must be non-empty and unique."};
      if (entry.contains("dtype") &&
          entry.at("dtype").get<std::string>() != "float32")
        throw std::invalid_argument{
            "Only float32 artifact tensors are supported."};
      auto const &shape = entry.at("shape");
      if (!shape.is_array() || shape.empty())
        throw std::invalid_argument{
            "Artifact tensor shapes must not be empty."};
      tensor.elementCount = 1u;
      for (auto const &extentValue : shape) {
        auto const extent64 = extentValue.get<std::uint64_t>();
        if (extent64 == 0u ||
            extent64 > std::numeric_limits<std::size_t>::max())
          throw std::invalid_argument{
              "Artifact tensor extents must be positive."};
        auto const extent = static_cast<std::size_t>(extent64);
        if (tensor.elementCount >
            detail::LearnedModelArtifactBuilder::maximumTensorElements / extent)
          throw std::invalid_argument{
              "The artifact tensor payload is too large."};
        tensor.elementCount *= extent;
        tensor.shape.push_back(extent);
      }
      if (totalElements >
          detail::LearnedModelArtifactBuilder::maximumTensorElements -
              tensor.elementCount)
        throw std::invalid_argument{
            "The artifact tensor payload is too large."};
      tensor.offset = totalElements;
      totalElements += tensor.elementCount;
      tensors.push_back(std::move(tensor));
    }

    auto payload = std::vector<char>(totalElements * sizeof(float));
    if (!input.read(payload.data(),
                    static_cast<std::streamsize>(payload.size())))
      return detail::invalidModelLoad(
          LearnedModelLoadStatus::invalidPayload,
          "The learned-model tensor payload is truncated.");
    if (input.peek() != std::ifstream::traits_type::eof())
      return detail::invalidModelLoad(
          LearnedModelLoadStatus::invalidPayload,
          "The learned-model artifact has trailing payload bytes.");
    auto values = std::vector<float>(totalElements);
    for (std::size_t index = 0u; index < totalElements; ++index) {
      auto const offset = index * sizeof(float);
      auto const bits = static_cast<std::uint32_t>(
                            static_cast<unsigned char>(payload[offset])) |
                        (static_cast<std::uint32_t>(
                             static_cast<unsigned char>(payload[offset + 1u]))
                         << 8u) |
                        (static_cast<std::uint32_t>(
                             static_cast<unsigned char>(payload[offset + 2u]))
                         << 16u) |
                        (static_cast<std::uint32_t>(
                             static_cast<unsigned char>(payload[offset + 3u]))
                         << 24u);
      values[index] = std::bit_cast<float>(bits);
      if (!std::isfinite(values[index]))
        return detail::invalidModelLoad(
            LearnedModelLoadStatus::invalidPayload,
            "The learned-model payload contains a non-finite value.");
    }

    auto artifact = detail::LearnedModelArtifactBuilder::build(
        std::move(metadataText), metadata, std::move(values),
        std::move(tensors));
    return {.status = LearnedModelLoadStatus::available,
            .message = {},
            .artifact = std::move(artifact)};
  } catch (std::exception const &error) {
    return detail::invalidModelLoad(LearnedModelLoadStatus::invalidMetadata,
                                    error.what());
  }
#endif
}

} // namespace alpakaTune
