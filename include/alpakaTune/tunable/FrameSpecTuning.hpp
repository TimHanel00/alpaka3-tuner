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

/** One independent FrameSpec frame-extent tuning entry. */
template <alpaka::onHost::concepts::FrameSpec T_FrameSpec, typename T_Values>
[[nodiscard]] auto tuneFrameExtent(T_FrameSpec const &, T_Values values) {
  return frameExtent(std::move(values));
}

/** One independent FrameSpec frame-count tuning entry. */
template <alpaka::onHost::concepts::FrameSpec T_FrameSpec, typename T_Values>
[[nodiscard]] auto tuneNumFrames(T_FrameSpec const &, T_Values values) {
  return numFrames(std::move(values));
}

/** One independent ThreadSpec block-count tuning entry. */
template <alpaka::onHost::concepts::ThreadSpec T_ThreadSpec, typename T_Values>
[[nodiscard]] auto tuneNumBlocks(T_ThreadSpec const &, T_Values values) {
  return numBlocks(std::move(values));
}

/** One independent ThreadSpec thread-count tuning entry. */
template <alpaka::onHost::concepts::ThreadSpec T_ThreadSpec, typename T_Values>
[[nodiscard]] auto tuneNumThreads(T_ThreadSpec const &, T_Values values) {
  return numThreads(std::move(values));
}

/** Keep the logical coverage of correlated FrameSpec candidates unchanged. */
template <alpaka::onHost::concepts::FrameSpec T_FrameSpec>
[[nodiscard]] auto preserveCoverage(T_FrameSpec const &frameSpec) {
  return alpakaTune::restrict(
      numFrames, frameExtent,
      [originalNumFrames = frameSpec.getNumFrames(),
       originalFrameExtents = frameSpec.getFrameExtents()](
          alpaka::concepts::VectorOrScalar auto const &candidateNumFrames,
          alpaka::concepts::VectorOrScalar auto const &candidateFrameExtent) {
        return detail::hasMatchingCoverage(
            candidateNumFrames, candidateFrameExtent, originalNumFrames,
            originalFrameExtents);
      });
}

/** Keep the logical coverage of correlated ThreadSpec candidates unchanged. */
template <alpaka::onHost::concepts::ThreadSpec T_ThreadSpec>
[[nodiscard]] auto preserveCoverage(T_ThreadSpec const &threadSpec) {
  return alpakaTune::restrict(
      numBlocks, numThreads,
      [originalNumBlocks = threadSpec.getNumBlocks(),
       originalNumThreads = threadSpec.getNumThreads()](
          alpaka::concepts::VectorOrScalar auto const &candidateNumBlocks,
          alpaka::concepts::VectorOrScalar auto const &candidateNumThreads) {
        return detail::hasMatchingCoverage(
            candidateNumBlocks, candidateNumThreads, originalNumBlocks,
            originalNumThreads);
      });
}

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
      throw std::invalid_argument{"Every frame-extent candidate must "
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
[[nodiscard]] auto frameCandidates(T_NumFrames const &originalNumFrames,
                                   T_FrameExtents const &originalFrameExtents,
                                   T_Values frameExtentValues) {
  using NumFrames = std::remove_cvref_t<T_NumFrames>;
  using RuntimeNumFrames = typename NumFrames::UniVec;
  auto numFrameValues = std::vector<RuntimeNumFrames>{};
  if constexpr (alpakaTune::detail::isRVals<T_Values>) {
    for (auto const &candidate : frameExtentValues.values())
      numFrameValues.push_back(matchingNumFrames(
          originalNumFrames, originalFrameExtents, candidate));
  } else if constexpr (alpakaTune::detail::isCTypes<T_Values>) {
    appendCompileTimeNumFrames<T_Values>(numFrameValues, originalNumFrames,
                                         originalFrameExtents);
  } else {
    static_assert(alpakaTune::detail::isRVals<T_Values> ||
                      alpakaTune::detail::isCTypes<T_Values>,
                  "Frame-extent candidates need RVals or CTypes.");
  }
  return std::pair{
      alpakaTune::RVals<RuntimeNumFrames>{std::move(numFrameValues)},
      std::move(frameExtentValues)};
}

/** Combine independent FrameSpec entries with one or more relations. */
template <typename T_FrameExtentTuning, typename T_NumFramesTuning,
          typename... T_Relations>
[[nodiscard]] auto makeFrameSpecTuning(T_FrameExtentTuning frameExtentTuning,
                                       T_NumFramesTuning numFramesTuning,
                                       T_Relations... relations) {
  using FrameExtentEntry = std::remove_cvref_t<T_FrameExtentTuning>;
  using NumFramesEntry = std::remove_cvref_t<T_NumFramesTuning>;
  static_assert(
      detail::sameName<FrameExtentEntry::name, detail::frameExtentName>,
      "makeFrameSpecTuning expects a tuneFrameExtent entry first.");
  static_assert(detail::sameName<NumFramesEntry::name, detail::numFramesName>,
                "makeFrameSpecTuning expects a tuneNumFrames entry second.");
  static_assert(sizeof...(T_Relations) > 0u,
                "Independent FrameSpec entries belong directly in a "
                "TunableBundle; makeFrameSpecTuning requires a relation.");
  return alpakaTune::constrain(
      alpakaTune::TunableBundle{std::move(frameExtentTuning),
                                std::move(numFramesTuning)},
      std::move(relations)...);
}

/** Default coverage-preserving FrameSpec tuning fragment. */
template <alpaka::onHost::concepts::FrameSpec T_FrameSpec>
[[nodiscard]] auto makeFrameSpecTuning(T_FrameSpec const &frameSpec) {
  using FrameExtents =
      typename std::remove_cvref_t<T_FrameSpec>::FrameExtentsVecType;
  if constexpr (alpaka::isCVector_v<FrameExtents>) {
    auto [numFrameValues, frameExtentValues] =
        frameCandidates(frameSpec.getNumFrames(), frameSpec.getFrameExtents(),
                        alpakaTune::CTypes<FrameExtents>{});
    return makeFrameSpecTuning(
        tuneFrameExtent(frameSpec, std::move(frameExtentValues)),
        tuneNumFrames(frameSpec, std::move(numFrameValues)),
        preserveCoverage(frameSpec));
  } else {
    auto [numFrameValues, frameExtentValues] =
        frameCandidates(frameSpec.getNumFrames(), frameSpec.getFrameExtents());
    return makeFrameSpecTuning(
        tuneFrameExtent(frameSpec, std::move(frameExtentValues)),
        tuneNumFrames(frameSpec, std::move(numFrameValues)),
        preserveCoverage(frameSpec));
  }
}

/** Combine independent ThreadSpec entries with one or more relations. */
template <typename T_NumBlocksTuning, typename T_NumThreadsTuning,
          typename... T_Relations>
[[nodiscard]] auto makeThreadSpecTuning(T_NumBlocksTuning numBlocksTuning,
                                        T_NumThreadsTuning numThreadsTuning,
                                        T_Relations... relations) {
  using NumBlocksEntry = std::remove_cvref_t<T_NumBlocksTuning>;
  using NumThreadsEntry = std::remove_cvref_t<T_NumThreadsTuning>;
  static_assert(detail::sameName<NumBlocksEntry::name, detail::numBlocksName>,
                "makeThreadSpecTuning expects a tuneNumBlocks entry first.");
  static_assert(detail::sameName<NumThreadsEntry::name, detail::numThreadsName>,
                "makeThreadSpecTuning expects a tuneNumThreads entry second.");
  static_assert(sizeof...(T_Relations) > 0u,
                "Independent ThreadSpec entries belong directly in a "
                "TunableBundle; makeThreadSpecTuning requires a relation.");
  return alpakaTune::constrain(
      alpakaTune::TunableBundle{std::move(numBlocksTuning),
                                std::move(numThreadsTuning)},
      std::move(relations)...);
}

/** Default coverage-preserving ThreadSpec tuning fragment. */
template <alpaka::onHost::concepts::ThreadSpec T_ThreadSpec>
[[nodiscard]] auto makeThreadSpecTuning(T_ThreadSpec const &threadSpec) {
  auto [blockValues, threadValues] =
      threadCandidates(threadSpec.getNumBlocks(), threadSpec.getNumThreads(),
                       !alpaka::isSeqExecutor(T_ThreadSpec::getExecutor()));
  return makeThreadSpecTuning(
      tuneNumBlocks(threadSpec, std::move(blockValues)),
      tuneNumThreads(threadSpec, std::move(threadValues)),
      preserveCoverage(threadSpec));
}

} // namespace alpakaTune
