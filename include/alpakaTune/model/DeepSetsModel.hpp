// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "alpakaTune/model/ModelArtifact.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace alpakaTune {

/** One deterministic ensemble prediction for a normalized candidate. */
struct LearnedModelPrediction {
  double logRuntimeSeconds{};
  double uncertainty{};
  std::vector<float> embedding;
};

/**
 * Framework-independent native inference for `deepsets_ensemble_v1`.
 *
 * The implementation intentionally uses only row-major float32 affine layers
 * and ReLU, so Python export and C++ deployment can be checked bit-for-bit
 * within normal floating-point tolerances without libtorch or ONNX Runtime.
 */
class NativeDeepSetsModel {
public:
  NativeDeepSetsModel(std::shared_ptr<LearnedModelArtifact const> artifact,
                      LearnedModelContextDescriptor descriptor)
      : m_artifact(std::move(artifact)), m_descriptor(std::move(descriptor)) {
    if (!m_artifact)
      throw std::invalid_argument{
          "Native learned inference needs a model artifact."};
    if (auto error = detail::validateLearnedContext(m_descriptor);
        !error.empty())
      throw std::invalid_argument{std::move(error)};
    validateFeatureContract();
    initialiseContextFeatures();
    initialiseMembers();
  }

  [[nodiscard]] auto predict(ParameterConfiguration const &configuration) const
      -> LearnedModelPrediction {
    validateParameterConfiguration(configuration, parameterSizes());
    auto tokenInput = dimensionTokens(configuration);
    auto memberPredictions = std::vector<double>{};
    memberPredictions.reserve(m_members.size());
    auto memberUncertainties = std::vector<double>{};
    memberUncertainties.reserve(m_members.size());
    auto meanEmbedding = std::vector<float>(m_artifact->embeddingSize(), 0.0f);

    for (auto const &member : m_members) {
      auto const embedding = inferEmbedding(member, tokenInput);
      auto const adapter = m_descriptor.deviceClass == LearnedDeviceClass::cpu
                               ? member.cpuAdapter
                               : member.gpuAdapter;
      auto const bias = m_descriptor.deviceClass == LearnedDeviceClass::cpu
                            ? member.cpuBias.front()
                            : member.gpuBias.front();
      auto prediction = static_cast<double>(bias);
      for (std::size_t index = 0u; index < embedding.size(); ++index) {
        prediction += static_cast<double>(adapter[index]) *
                      static_cast<double>(embedding[index]);
        meanEmbedding[index] +=
            embedding[index] / static_cast<float>(m_members.size());
      }
      if (!std::isfinite(prediction))
        throw std::runtime_error{
            "Native learned inference produced a non-finite prediction."};
      memberPredictions.push_back(prediction);
      auto const uncertaintyAdapter =
          m_descriptor.deviceClass == LearnedDeviceClass::cpu
              ? member.cpuUncertainty
              : member.gpuUncertainty;
      auto const uncertaintyBias =
          m_descriptor.deviceClass == LearnedDeviceClass::cpu
              ? member.cpuUncertaintyBias
              : member.gpuUncertaintyBias;
      if (!uncertaintyAdapter.empty()) {
        auto raw = static_cast<double>(uncertaintyBias.front());
        for (std::size_t index = 0u; index < embedding.size(); ++index)
          raw += static_cast<double>(uncertaintyAdapter[index]) *
                 static_cast<double>(embedding[index]);
        memberUncertainties.push_back(softplus(raw));
      }
    }

    auto const mean = std::accumulate(memberPredictions.begin(),
                                      memberPredictions.end(), 0.0) /
                      static_cast<double>(memberPredictions.size());
    auto variance = 0.0;
    for (auto const prediction : memberPredictions) {
      auto const difference = prediction - mean;
      variance += difference * difference;
    }
    variance /= static_cast<double>(memberPredictions.size());
    auto uncertainty = std::sqrt(std::max(variance, 0.0));
    if (memberPredictions.size() == 1u && !memberUncertainties.empty())
      uncertainty = memberUncertainties.front();
    return {.logRuntimeSeconds = mean,
            .uncertainty = uncertainty,
            .embedding = std::move(meanEmbedding)};
  }

  /** Score a homogeneous batch without changing scalar inference semantics. */
  [[nodiscard]] auto
  predictBatch(std::span<ParameterConfiguration const> configurations) const
      -> std::vector<LearnedModelPrediction> {
    auto predictions = std::vector<LearnedModelPrediction>{};
    predictions.reserve(configurations.size());
    for (auto const &configuration : configurations)
      predictions.push_back(predict(configuration));
    return predictions;
  }

  [[nodiscard]] auto parameterSizes() const noexcept
      -> std::span<std::size_t const> {
    return m_parameterSizes;
  }

  [[nodiscard]] auto descriptor() const noexcept
      -> LearnedModelContextDescriptor const & {
    return m_descriptor;
  }

  [[nodiscard]] auto artifact() const noexcept -> LearnedModelArtifact const & {
    return *m_artifact;
  }

private:
  struct Member {
    std::span<float const> token0Weight;
    std::span<float const> token0Bias;
    std::span<float const> token2Weight;
    std::span<float const> token2Bias;
    std::span<float const> context0Weight;
    std::span<float const> context0Bias;
    std::span<float const> context2Weight;
    std::span<float const> context2Bias;
    std::span<float const> cpuAdapter;
    std::span<float const> cpuBias;
    std::span<float const> gpuAdapter;
    std::span<float const> gpuBias;
    std::span<float const> cpuUncertainty;
    std::span<float const> cpuUncertaintyBias;
    std::span<float const> gpuUncertainty;
    std::span<float const> gpuUncertaintyBias;
  };

  void validateFeatureContract() const {
    if (m_artifact->architecture() != detail::learnedArchitectureName)
      throw std::invalid_argument{
          "The model artifact has an unsupported architecture."};
    if (m_artifact->featureSchema() !=
        LearnedModelContextDescriptor::featureSchemaVersion)
      throw std::invalid_argument{
          "The model and native feature schemas differ."};
    if (m_artifact->dimensionFeatureCount() !=
        detail::learnedDimensionFeatureCount)
      throw std::invalid_argument{
          "The model has the wrong dimension feature count."};
    auto const names = m_artifact->dimensionFeatureNames();
    for (std::size_t index = 0u; index < names.size(); ++index) {
      if (names[index] != detail::learnedDimensionFeatureNames[index])
        throw std::invalid_argument{
            "The model dimension feature order is incompatible."};
    }
    if (m_artifact->deviceClassCount() != 2u)
      throw std::invalid_argument{
          "The model must contain CPU and GPU adapters."};
    if (m_artifact->adapterFeatureCount() > m_artifact->embeddingSize())
      throw std::invalid_argument{
          "The online-adapter feature count is invalid."};
  }

  void initialiseContextFeatures() {
    auto byName = std::unordered_map<std::string_view, float>{};
    byName.reserve(m_descriptor.contextFeatures.size());
    for (auto const &feature : m_descriptor.contextFeatures)
      byName.emplace(feature.name, feature.value);
    auto const names = m_artifact->contextFeatureNames();
    m_contextFeatures.resize(names.size());
    for (std::size_t index = 0u; index < names.size(); ++index) {
      auto const found = byName.find(names[index]);
      auto const raw = found == byName.end() ? 0.0f : found->second;
      m_contextFeatures[index] = (raw - m_artifact->contextMean()[index]) /
                                 m_artifact->contextScale()[index];
    }
    m_parameterSizes.reserve(m_descriptor.dimensions.size());
    m_concreteMinimum.reserve(m_descriptor.dimensions.size());
    m_concreteRange.reserve(m_descriptor.dimensions.size());
    for (auto const &dimension : m_descriptor.dimensions) {
      m_parameterSizes.push_back(dimension.cardinality);
      if (dimension.concreteValues.empty()) {
        m_concreteMinimum.push_back(0.0f);
        m_concreteRange.push_back(0.0f);
      } else {
        auto const [minimum, maximum] =
            std::ranges::minmax(dimension.concreteValues);
        m_concreteMinimum.push_back(minimum);
        m_concreteRange.push_back(maximum - minimum);
      }
    }
  }

  [[nodiscard]] auto requireTensor(std::string const &name,
                                   std::span<std::size_t const> shape) const
      -> std::span<float const> {
    auto const descriptor = m_artifact->tensorDescriptor(name);
    auto const values = m_artifact->tensor(name);
    if (descriptor == nullptr || !values ||
        !std::ranges::equal(descriptor->shape, shape))
      throw std::invalid_argument{
          "The model tensor has a missing or invalid shape: " + name};
    return *values;
  }

  [[nodiscard]] auto optionalTensor(std::string const &name,
                                    std::span<std::size_t const> shape) const
      -> std::span<float const> {
    auto const descriptor = m_artifact->tensorDescriptor(name);
    if (descriptor == nullptr)
      return {};
    auto const values = m_artifact->tensor(name);
    if (!values || !std::ranges::equal(descriptor->shape, shape))
      throw std::invalid_argument{
          "The optional model tensor has an invalid shape: " + name};
    return *values;
  }

  void initialiseMembers() {
    auto const hiddenSizes = m_artifact->tokenHiddenSizes();
    if (hiddenSizes.size() != 2u)
      throw std::invalid_argument{
          "The model must declare two token hidden widths."};
    auto const token0 = hiddenSizes[0];
    auto const token2 = hiddenSizes[1];
    auto const dimension = m_artifact->dimensionFeatureCount();
    auto const embedding = m_artifact->embeddingSize();
    auto const contextInput = 2u * token2 + m_artifact->contextFeatureCount();
    auto const token0WeightShape = std::vector<std::size_t>{token0, dimension};
    auto const token0BiasShape = std::vector<std::size_t>{token0};
    auto const token2WeightShape = std::vector<std::size_t>{token2, token0};
    auto const token2BiasShape = std::vector<std::size_t>{token2};
    auto const context0WeightShape =
        std::vector<std::size_t>{embedding, contextInput};
    auto const embeddingBiasShape = std::vector<std::size_t>{embedding};
    auto const context2WeightShape =
        std::vector<std::size_t>{embedding, embedding};
    auto const adapterWeightShape = std::vector<std::size_t>{1u, embedding};
    auto const adapterBiasShape = std::vector<std::size_t>{1u};
    m_members.reserve(m_artifact->ensembleSize());
    for (std::size_t index = 0u; index < m_artifact->ensembleSize(); ++index) {
      auto const prefix = std::string{"members."} + std::to_string(index) + '.';
      m_members.push_back(Member{
          .token0Weight =
              requireTensor(prefix + "token.0.weight", token0WeightShape),
          .token0Bias = requireTensor(prefix + "token.0.bias", token0BiasShape),
          .token2Weight =
              requireTensor(prefix + "token.2.weight", token2WeightShape),
          .token2Bias = requireTensor(prefix + "token.2.bias", token2BiasShape),
          .context0Weight =
              requireTensor(prefix + "context.0.weight", context0WeightShape),
          .context0Bias =
              requireTensor(prefix + "context.0.bias", embeddingBiasShape),
          .context2Weight =
              requireTensor(prefix + "context.2.weight", context2WeightShape),
          .context2Bias =
              requireTensor(prefix + "context.2.bias", embeddingBiasShape),
          .cpuAdapter =
              requireTensor(prefix + "adapters.cpu.weight", adapterWeightShape),
          .cpuBias =
              requireTensor(prefix + "adapters.cpu.bias", adapterBiasShape),
          .gpuAdapter =
              requireTensor(prefix + "adapters.gpu.weight", adapterWeightShape),
          .gpuBias =
              requireTensor(prefix + "adapters.gpu.bias", adapterBiasShape),
          .cpuUncertainty = optionalTensor(prefix + "uncertainty.cpu.weight",
                                           adapterWeightShape),
          .cpuUncertaintyBias =
              optionalTensor(prefix + "uncertainty.cpu.bias", adapterBiasShape),
          .gpuUncertainty = optionalTensor(prefix + "uncertainty.gpu.weight",
                                           adapterWeightShape),
          .gpuUncertaintyBias =
              optionalTensor(prefix + "uncertainty.gpu.bias", adapterBiasShape),
      });
      auto const &member = m_members.back();
      auto const hasAnyUncertainty = !member.cpuUncertainty.empty() ||
                                     !member.cpuUncertaintyBias.empty() ||
                                     !member.gpuUncertainty.empty() ||
                                     !member.gpuUncertaintyBias.empty();
      auto const hasAllUncertainty = !member.cpuUncertainty.empty() &&
                                     !member.cpuUncertaintyBias.empty() &&
                                     !member.gpuUncertainty.empty() &&
                                     !member.gpuUncertaintyBias.empty();
      if (hasAnyUncertainty != hasAllUncertainty ||
          (!m_artifact->uncertaintyHead().empty() && !hasAllUncertainty))
        throw std::invalid_argument{"The model uncertainty head must contain "
                                    "complete CPU and GPU tensors."};
    }
  }

  [[nodiscard]] auto
  dimensionTokens(ParameterConfiguration const &configuration) const
      -> std::vector<float> {
    auto const width = m_artifact->dimensionFeatureCount();
    auto result = std::vector<float>(configuration.size() * width);
    for (std::size_t dimensionIndex = 0u; dimensionIndex < configuration.size();
         ++dimensionIndex) {
      auto const &dimension = m_descriptor.dimensions[dimensionIndex];
      auto const coordinate = configuration[dimensionIndex];
      auto const concreteIndex =
          dimension.cardinality == 1u
              ? 0u
              : static_cast<std::size_t>(std::lround(
                    static_cast<double>(coordinate) *
                    static_cast<double>(dimension.cardinality - 1u)));
      auto raw = std::array<float, detail::learnedDimensionFeatureCount>{};
      raw[0] = coordinate;
      if (dimension.concreteValues.empty()) {
        raw[1] = coordinate;
      } else {
        raw[1] = m_concreteRange[dimensionIndex] == 0.0f
                     ? 0.0f
                     : (dimension.concreteValues.at(concreteIndex) -
                        m_concreteMinimum[dimensionIndex]) /
                           m_concreteRange[dimensionIndex];
      }
      raw[2] = std::log1p(static_cast<float>(dimension.cardinality));
      raw[3] =
          m_descriptor.dimensions.size() == 1u
              ? 0.0f
              : static_cast<float>(dimensionIndex) /
                    static_cast<float>(m_descriptor.dimensions.size() - 1u);
      raw[4] = dimension.vectorArity == 1u
                   ? 0.0f
                   : static_cast<float>(dimension.componentIndex) /
                         static_cast<float>(dimension.vectorArity - 1u);
      raw[5] = std::log1p(static_cast<float>(dimension.vectorArity));
      raw[6u + static_cast<std::size_t>(dimension.kind)] = 1.0f;
      auto const hashes = detail::signedHashFeatures(
          dimension.name, detail::learnedDimensionNameHashBuckets);
      std::ranges::copy(hashes, raw.begin() + 10u);
      for (std::size_t feature = 0u; feature < width; ++feature) {
        result[dimensionIndex * width + feature] =
            (raw[feature] - m_artifact->dimensionMean()[feature]) /
            m_artifact->dimensionScale()[feature];
      }
    }
    return result;
  }

  [[nodiscard]] static auto affineRelu(std::span<float const> input,
                                       std::span<float const> weight,
                                       std::span<float const> bias)
      -> std::vector<float> {
    auto output = std::vector<float>(bias.size());
    affineReluInto(input, weight, bias, output);
    return output;
  }

  static void affineReluInto(std::span<float const> input,
                             std::span<float const> weight,
                             std::span<float const> bias,
                             std::span<float> output) {
    for (std::size_t row = 0u; row < bias.size(); ++row) {
      auto value = bias[row];
      for (std::size_t column = 0u; column < input.size(); ++column)
        value += weight[row * input.size() + column] * input[column];
      output[row] = std::max(value, 0.0f);
    }
  }

  [[nodiscard]] auto inferEmbedding(Member const &member,
                                    std::span<float const> tokens) const
      -> std::vector<float> {
    auto const dimensionCount = m_descriptor.dimensions.size();
    auto const featureCount = m_artifact->dimensionFeatureCount();
    auto const hidden = m_artifact->tokenHiddenSizes().back();
    auto meanPool = std::vector<float>(hidden, 0.0f);
    auto maxPool =
        std::vector<float>(hidden, -std::numeric_limits<float>::infinity());
    auto tokenFirst =
        std::vector<float>(m_artifact->tokenHiddenSizes().front());
    auto tokenSecond = std::vector<float>(hidden);
    for (std::size_t dimension = 0u; dimension < dimensionCount; ++dimension) {
      auto token = tokens.subspan(dimension * featureCount, featureCount);
      affineReluInto(token, member.token0Weight, member.token0Bias, tokenFirst);
      affineReluInto(tokenFirst, member.token2Weight, member.token2Bias,
                     tokenSecond);
      for (std::size_t index = 0u; index < hidden; ++index) {
        meanPool[index] +=
            tokenSecond[index] / static_cast<float>(dimensionCount);
        maxPool[index] = std::max(maxPool[index], tokenSecond[index]);
      }
    }
    auto contextInput = std::vector<float>{};
    contextInput.reserve(2u * hidden + m_contextFeatures.size());
    contextInput.insert(contextInput.end(), meanPool.begin(), meanPool.end());
    contextInput.insert(contextInput.end(), maxPool.begin(), maxPool.end());
    contextInput.insert(contextInput.end(), m_contextFeatures.begin(),
                        m_contextFeatures.end());
    auto first =
        affineRelu(contextInput, member.context0Weight, member.context0Bias);
    return affineRelu(first, member.context2Weight, member.context2Bias);
  }

  [[nodiscard]] static auto softplus(double value) noexcept -> double {
    if (value > 20.0)
      return value;
    if (value < -20.0)
      return std::exp(value);
    return std::log1p(std::exp(value));
  }

  std::shared_ptr<LearnedModelArtifact const> m_artifact;
  LearnedModelContextDescriptor m_descriptor;
  std::vector<std::size_t> m_parameterSizes;
  std::vector<float> m_concreteMinimum;
  std::vector<float> m_concreteRange;
  std::vector<float> m_contextFeatures;
  std::vector<Member> m_members;
};

} // namespace alpakaTune
