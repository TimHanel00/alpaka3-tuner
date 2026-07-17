// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpakaTune/core/Tuner.hpp>
#include <alpakaTune/tunable/Tunables.hpp>

#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace alpakaTune {
namespace detail {
template <alpaka::concepts::VectorOrScalar T_First,
          alpaka::concepts::VectorOrScalar T_Second,
          alpaka::concepts::VectorOrScalar T_ReferenceFirst,
          alpaka::concepts::VectorOrScalar T_ReferenceSecond>
[[nodiscard]] auto hasMatchingCoverage(T_First const &first,
                                       T_Second const &second,
                                       T_ReferenceFirst const &referenceFirst,
                                       T_ReferenceSecond const &referenceSecond)
    -> bool {
  if constexpr (alpaka::concepts::Vector<T_First> &&
                alpaka::concepts::Vector<T_Second>) {
    if constexpr (T_First::dim() != T_Second::dim())
      return false;
    for (std::size_t dimension = 0u; dimension < T_First::dim(); ++dimension)
      if (first[dimension] * second[dimension] !=
          referenceFirst[dimension] * referenceSecond[dimension])
        return false;
    return true;
  } else
    return first * second == referenceFirst * referenceSecond;
}
} // namespace detail

inline constexpr auto frameExtent = ALPAKA_TUNE_TUNABLE("frameExtent");
inline constexpr auto numFrames = ALPAKA_TUNE_TUNABLE("numFrames");
inline constexpr auto numBlocks = ALPAKA_TUNE_TUNABLE("numBlocks");
inline constexpr auto numThreads = ALPAKA_TUNE_TUNABLE("numThreads");

/** Candidate representations for the logical extent of a FrameSpec. */
template <typename T_Values> struct FrameExtentTuning {
  T_Values values;
};

template <typename T_Values>
FrameExtentTuning(T_Values) -> FrameExtentTuning<T_Values>;

template <typename T_Values> struct NumFramesTuning {
  T_Values values;
};

template <typename T_Values>
NumFramesTuning(T_Values) -> NumFramesTuning<T_Values>;

template <typename T_NumFrames, typename T_FrameExtents>
[[nodiscard]] auto frameCandidates(T_NumFrames const &numFramesValue,
                                   T_FrameExtents const &frameExtentsValue) {
  using NumFrames = std::remove_cvref_t<T_NumFrames>;
  using FrameExtents = std::remove_cvref_t<T_FrameExtents>;
  using RuntimeNumFrames = typename NumFrames::UniVec;
  using RuntimeFrameExtents = typename FrameExtents::UniVec;

  RuntimeNumFrames const originalNumFrames{numFramesValue};
  RuntimeFrameExtents const originalFrameExtents{frameExtentsValue};
  auto alternateNumFrames = originalNumFrames;
  auto alternateFrameExtents = originalFrameExtents;
  auto hasAlternate = false;
  for (std::size_t dimension = 0u; dimension < RuntimeFrameExtents::dim();
       ++dimension) {
    if (alternateFrameExtents[dimension] > 1u &&
        alternateFrameExtents[dimension] % 2u == 0u) {
      alternateNumFrames[dimension] *= 2u;
      alternateFrameExtents[dimension] /= 2u;
      hasAlternate = true;
    }
  }

  auto numFrameValues = std::vector<RuntimeNumFrames>{originalNumFrames};
  auto frameExtentValues =
      std::vector<RuntimeFrameExtents>{originalFrameExtents};
  if (hasAlternate) {
    numFrameValues.push_back(alternateNumFrames);
    frameExtentValues.push_back(alternateFrameExtents);
  }
  return std::pair{
      alpakaTune::RVals<RuntimeNumFrames>{std::move(numFrameValues)},
      alpakaTune::RVals<RuntimeFrameExtents>{std::move(frameExtentValues)}};
}

template <typename T_NumBlocks, typename T_NumThreads>
[[nodiscard]] auto threadCandidates(T_NumBlocks const &numBlocksValue,
                                    T_NumThreads const &numThreadsValue,
                                    bool const allowAlternate) {
  using NumBlocks = std::remove_cvref_t<T_NumBlocks>;
  using NumThreads = std::remove_cvref_t<T_NumThreads>;
  using RuntimeNumBlocks = typename NumBlocks::UniVec;
  using RuntimeNumThreads = typename NumThreads::UniVec;

  RuntimeNumBlocks const originalNumBlocks{numBlocksValue};
  RuntimeNumThreads const originalNumThreads{numThreadsValue};
  auto alternateNumBlocks = originalNumBlocks;
  auto alternateNumThreads = originalNumThreads;
  auto hasAlternate = false;
  for (std::size_t dimension = 0u; dimension < RuntimeNumBlocks::dim();
       ++dimension) {
    if (allowAlternate && alternateNumBlocks[dimension] > 1u &&
        alternateNumBlocks[dimension] % 2u == 0u) {
      alternateNumBlocks[dimension] /= 2u;
      alternateNumThreads[dimension] *= 2u;
      hasAlternate = true;
    }
  }

  auto blockValues = std::vector<RuntimeNumBlocks>{originalNumBlocks};
  auto threadValues = std::vector<RuntimeNumThreads>{originalNumThreads};
  if (hasAlternate) {
    blockValues.push_back(alternateNumBlocks);
    threadValues.push_back(alternateNumThreads);
  }
  return std::pair{
      alpakaTune::RVals<RuntimeNumBlocks>{std::move(blockValues)},
      alpakaTune::RVals<RuntimeNumThreads>{std::move(threadValues)}};
}

template <typename T_NumFrames, typename T_FrameExtents,
          typename T_CandidateFrameExtent>
[[nodiscard]] auto
matchingNumFrames(T_NumFrames const &originalNumFramesValue,
                  T_FrameExtents const &originalFrameExtentsValue,
                  T_CandidateFrameExtent const &candidateFrameExtent) {
  using NumFrames = std::remove_cvref_t<T_NumFrames>;
  using FrameExtents = std::remove_cvref_t<T_FrameExtents>;
  using RuntimeNumFrames = typename NumFrames::UniVec;
  using RuntimeFrameExtents = typename FrameExtents::UniVec;

  RuntimeNumFrames const originalNumFrames{originalNumFramesValue};
  RuntimeFrameExtents const originalFrameExtents{originalFrameExtentsValue};
  RuntimeFrameExtents const candidateExtents{candidateFrameExtent};
  auto result = originalNumFrames;
  for (std::size_t dimension = 0u; dimension < RuntimeFrameExtents::dim();
       ++dimension) {
    auto const coverage =
        originalNumFrames[dimension] * originalFrameExtents[dimension];
    if (candidateExtents[dimension] == 0u ||
        coverage % candidateExtents[dimension] != 0u)
      throw std::invalid_argument{"Every FrameExtentTuning candidate must "
                                  "divide the original logical extent."};
    result[dimension] = coverage / candidateExtents[dimension];
  }
  return result;
}

template <typename T_Values, typename T_NumFrames, typename T_FrameExtents,
          std::size_t T_Index = 0u>
void appendCompileTimeNumFrames(
    std::vector<typename std::remove_cvref_t<T_NumFrames>::UniVec> &values,
    T_NumFrames const &originalNumFrames,
    T_FrameExtents const &originalFrameExtents) {
  if constexpr (T_Index < T_Values::size) {
    using Candidate = std::tuple_element_t<T_Index, typename T_Values::values>;
    values.push_back(
        matchingNumFrames(originalNumFrames, originalFrameExtents,
                          alpakaTune::detail::launchValue<Candidate>()));
    appendCompileTimeNumFrames<T_Values, T_NumFrames, T_FrameExtents,
                               T_Index + 1u>(values, originalNumFrames,
                                             originalFrameExtents);
  }
}

template <typename T_NumFrames, typename T_FrameExtents, typename T_Values>
[[nodiscard]] auto
frameCandidates(T_NumFrames const &originalNumFrames,
                T_FrameExtents const &originalFrameExtents,
                FrameExtentTuning<T_Values> frameExtentTuning) {
  using NumFrames = std::remove_cvref_t<T_NumFrames>;
  using RuntimeNumFrames = typename NumFrames::UniVec;
  auto numFrameValues = std::vector<RuntimeNumFrames>{};
  if constexpr (alpakaTune::detail::isRVals<T_Values>) {
    for (auto const &candidate : frameExtentTuning.values.values())
      numFrameValues.push_back(matchingNumFrames(
          originalNumFrames, originalFrameExtents, candidate));
  } else if constexpr (alpakaTune::detail::isCTypes<T_Values>) {
    appendCompileTimeNumFrames<T_Values>(numFrameValues, originalNumFrames,
                                         originalFrameExtents);
  } else {
    static_assert(alpakaTune::detail::isRVals<T_Values> ||
                      alpakaTune::detail::isCTypes<T_Values>,
                  "FrameExtentTuning needs RVals or CTypes.");
  }
  return std::pair{
      alpakaTune::RVals<RuntimeNumFrames>{std::move(numFrameValues)},
      std::move(frameExtentTuning.values)};
}

template <typename T_Device, alpaka::onHost::concepts::FrameSpec T_FrameSpec,
          typename T_Values>
[[nodiscard]] auto makeTuner(TunerConfig config, T_Device const &device,
                             T_FrameSpec const &frameSpec,
                             std::string_view identity,
                             FrameExtentTuning<T_Values> frameExtentTuning) {
  auto [numFrameValues, frameExtentValues] =
      frameCandidates(frameSpec.getNumFrames(), frameSpec.getFrameExtents(),
                      std::move(frameExtentTuning));
  auto const tunables = alpakaTune::constrain(
      alpakaTune::TunableBundle{
          alpakaTune::named(numFrames, std::move(numFrameValues)),
          alpakaTune::named(frameExtent, std::move(frameExtentValues))},
      alpakaTune::restrict(
          numFrames, frameExtent,
          [originalNumFrames = frameSpec.getNumFrames(),
           originalFrameExtents = frameSpec.getFrameExtents()](
              alpaka::concepts::VectorOrScalar auto const &candidateNumFrames,
              alpaka::concepts::VectorOrScalar auto const
                  &candidateFrameExtent) {
            return detail::hasMatchingCoverage(
                candidateNumFrames, candidateFrameExtent, originalNumFrames,
                originalFrameExtents);
          }));
  return alpakaTune::makeTuner(std::move(config), tunables, device,
                               frameSpec.getExecutor(), identity);
}

template <typename T_Device, alpaka::onHost::concepts::FrameSpec T_FrameSpec,
          typename T_Values>
[[nodiscard]] auto makeTuner(T_Device const &device,
                             T_FrameSpec const &frameSpec,
                             std::string_view identity,
                             FrameExtentTuning<T_Values> frameExtentTuning) {
  return makeTuner(tunerConfig(), device, frameSpec, identity,
                   std::move(frameExtentTuning));
}

template <typename T_Device, alpaka::onHost::concepts::FrameSpec T_FrameSpec>
[[nodiscard]] auto makeTuner(TunerConfig config, T_Device const &device,
                             T_FrameSpec const &frameSpec,
                             std::string_view identity) {
  using FrameExtents =
      typename std::remove_cvref_t<T_FrameSpec>::FrameExtentsVecType;
  if constexpr (alpaka::isCVector_v<FrameExtents>) {
    return makeTuner(std::move(config), device, frameSpec, identity,
                     FrameExtentTuning{alpakaTune::CTypes<FrameExtents>{}});
  } else {
    auto [numFrameValues, frameExtentValues] =
        frameCandidates(frameSpec.getNumFrames(), frameSpec.getFrameExtents());
    auto const tunables = alpakaTune::constrain(
        alpakaTune::TunableBundle{
            alpakaTune::named(numFrames, std::move(numFrameValues)),
            alpakaTune::named(frameExtent, std::move(frameExtentValues))},
        alpakaTune::restrict(
            numFrames, frameExtent,
            [originalNumFrames = frameSpec.getNumFrames(),
             originalFrameExtents = frameSpec.getFrameExtents()](
                alpaka::concepts::VectorOrScalar auto const &candidateNumFrames,
                alpaka::concepts::VectorOrScalar auto const
                    &candidateFrameExtent) {
              return detail::hasMatchingCoverage(
                  candidateNumFrames, candidateFrameExtent, originalNumFrames,
                  originalFrameExtents);
            }));
    return alpakaTune::makeTuner(std::move(config), tunables, device,
                                 frameSpec.getExecutor(), identity);
  }
}

template <typename T_Device, alpaka::onHost::concepts::FrameSpec T_FrameSpec>
[[nodiscard]] auto makeTuner(T_Device const &device,
                             T_FrameSpec const &frameSpec,
                             std::string_view identity) {
  return makeTuner(tunerConfig(), device, frameSpec, identity);
}

template <typename T_Device, alpaka::onHost::concepts::FrameSpec T_FrameSpec,
          typename T_NumFrameValues, typename T_FrameExtentValues>
[[nodiscard]] auto
makeTuner(TunerConfig config, T_Device const &device,
          T_FrameSpec const &threadSpec, std::string_view identity,
          NumFramesTuning<T_NumFrameValues> numFramesTuning,
          FrameExtentTuning<T_FrameExtentValues> frameExtentTuning) {
  auto tunables = alpakaTune::TunableBundle{
      numFrames(std::move(numFramesTuning.values)),
      frameExtent(std::move(frameExtentTuning.values))};
  return alpakaTune::makeTuner(std::move(config), std::move(tunables), device,
                               threadSpec.getExecutor(), identity);
}

template <typename T_Device, alpaka::onHost::concepts::FrameSpec T_FrameSpec,
          typename T_NumFrameValues, typename T_FrameExtentValues>
[[nodiscard]] auto
makeTuner(T_Device const &device, T_FrameSpec const &frameSpec,
          std::string_view identity,
          NumFramesTuning<T_NumFrameValues> numFramesTuning,
          FrameExtentTuning<T_FrameExtentValues> frameExtentTuning) {
  return makeTuner(tunerConfig(), device, frameSpec, identity,
                   std::move(numFramesTuning), std::move(frameExtentTuning));
}

template <typename T_Device, alpaka::onHost::concepts::ThreadSpec T_ThreadSpec>
[[nodiscard]] auto makeTuner(TunerConfig config, T_Device const &device,
                             T_ThreadSpec const &threadSpec,
                             std::string_view identity) {
  auto [blockValues, threadValues] =
      threadCandidates(threadSpec.getNumBlocks(), threadSpec.getNumThreads(),
                       !alpaka::isSeqExecutor(T_ThreadSpec::getExecutor()));
  auto const tunables = alpakaTune::constrain(
      alpakaTune::TunableBundle{
          alpakaTune::named(numBlocks, std::move(blockValues)),
          alpakaTune::named(numThreads, std::move(threadValues))},
      alpakaTune::restrict(
          numBlocks, numThreads,
          [originalNumBlocks = threadSpec.getNumBlocks(),
           originalNumThreads = threadSpec.getNumThreads()](
              alpaka::concepts::VectorOrScalar auto const &candidateNumBlocks,
              alpaka::concepts::VectorOrScalar auto const
                  &candidateNumThreads) {
            return detail::hasMatchingCoverage(
                candidateNumBlocks, candidateNumThreads, originalNumBlocks,
                originalNumThreads);
          }));
  return alpakaTune::makeTuner(std::move(config), tunables, device,
                               threadSpec.getExecutor(), identity);
}

template <typename T_Device, alpaka::onHost::concepts::ThreadSpec T_ThreadSpec>
[[nodiscard]] auto makeTuner(T_Device const &device,
                             T_ThreadSpec const &threadSpec,
                             std::string_view identity) {
  return makeTuner(tunerConfig(), device, threadSpec, identity);
}

} // namespace alpakaTune
