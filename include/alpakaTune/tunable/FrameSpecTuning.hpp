// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpakaTune/core/Tuner.hpp>
#include <alpakaTune/tunable/Tunables.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace alpakaTune {
namespace detail {
inline constexpr auto defaultFrameElementCounts =
    std::array<std::size_t, 6u>{32u, 64u, 128u, 256u, 512u, 1024u};

[[nodiscard]] auto
isDefaultFrameExtent(alpaka::concepts::Vector auto const &candidate) -> bool {
  auto elements = std::size_t{1u};
  for (std::size_t dimension = 1u; dimension < ALPAKA_TYPEOF(candidate)::dim();
       ++dimension)
    if (candidate[dimension - 1u] > candidate[dimension])
      return false;
  for (std::size_t dimension = 0u; dimension < ALPAKA_TYPEOF(candidate)::dim();
       ++dimension) {
    auto const component = static_cast<std::size_t>(candidate[dimension]);
    if (component == 0u ||
        elements > std::numeric_limits<std::size_t>::max() / component)
      return false;
    elements *= component;
  }
  return std::ranges::find(defaultFrameElementCounts, elements) !=
         defaultFrameElementCounts.end();
}

void appendUnique(auto &values, auto const &candidate) {
  if (std::ranges::find(values, candidate) == values.end())
    values.push_back(candidate);
}

[[nodiscard]] auto splitCandidates(std::integral auto upperLimit,
                                   std::size_t refinementLevels) {
  if (upperLimit < ALPAKA_TYPEOF(upperLimit){1})
    throw std::invalid_argument{
        "A default numFrames upper limit must be positive."};

  auto splits = std::vector{upperLimit};
  for (auto value = upperLimit;;) {
    if (value == ALPAKA_TYPEOF(upperLimit){1})
      break;
    value /= 2u;
    splits.push_back(value);
  }
  std::ranges::reverse(splits);

  auto values = splits;
  for (std::size_t level = 0u; level < refinementLevels; ++level) {
    auto const current = values;
    auto added = false;
    for (std::size_t index = 1u; index < current.size(); ++index) {
      auto const lower = current[index - 1u];
      auto const upper = current[index];
      ALPAKA_TYPEOF(upperLimit)
      const midpoint = lower + (upper - lower) / ALPAKA_TYPEOF(upperLimit){2};
      if (midpoint != lower && midpoint != upper) {
        values.push_back(midpoint);
        added = true;
      }
    }
    if (!added)
      break;
    std::ranges::sort(values);
    auto const uniqueEnd = std::ranges::unique(values).begin();
    values.erase(uniqueEnd, values.end());
  }
  return values;
}

[[nodiscard]] auto productDoesNotExceed(std::integral auto left,
                                        std::integral auto right,
                                        std::integral auto referenceLeft,
                                        std::integral auto referenceRight)
    -> bool {
  if (left <= ALPAKA_TYPEOF(left){} || right <= ALPAKA_TYPEOF(right){})
    return false;
  if (referenceLeft <= ALPAKA_TYPEOF(referenceLeft){} ||
      referenceRight <= ALPAKA_TYPEOF(referenceRight){})
    throw std::invalid_argument{
        "The original FrameSpec coverage must be positive."};
  using Common =
      std::common_type_t<ALPAKA_TYPEOF(left), ALPAKA_TYPEOF(right),
                         ALPAKA_TYPEOF(referenceLeft),
                         ALPAKA_TYPEOF(referenceRight), std::uintmax_t>;
  auto const candidateLeft = static_cast<Common>(left);
  auto const candidateRight = static_cast<Common>(right);
  auto const originalLeft = static_cast<Common>(referenceLeft);
  auto const originalRight = static_cast<Common>(referenceRight);
  auto const maximum = std::numeric_limits<Common>::max();
  if (originalLeft != Common{} && originalRight > maximum / originalLeft)
    throw std::overflow_error{
        "The original FrameSpec coverage exceeds the comparison type."};
  if (candidateLeft != Common{} && candidateRight > maximum / candidateLeft)
    return false;
  return candidateLeft * candidateRight <= originalLeft * originalRight;
}

[[nodiscard]] auto hasMatchingCoverage(
    alpaka::concepts::VectorOrScalar auto const &first,
    alpaka::concepts::VectorOrScalar auto const &second,
    alpaka::concepts::VectorOrScalar auto const &referenceFirst,
    alpaka::concepts::VectorOrScalar auto const &referenceSecond) -> bool {
  if constexpr (alpaka::isVector_v<ALPAKA_TYPEOF(first)> &&
                alpaka::isVector_v<ALPAKA_TYPEOF(second)>) {
    if constexpr (ALPAKA_TYPEOF(first)::dim() != ALPAKA_TYPEOF(second)::dim())
      return false;
    for (std::size_t dimension = 0u; dimension < ALPAKA_TYPEOF(first)::dim();
         ++dimension)
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
[[nodiscard]] auto
tuneFrameExtent(alpaka::onHost::concepts::FrameSpec auto const &, auto values) {
  return frameExtent(std::move(values));
}

/** Select how an explicit multidimensional frame-extent list is combined. */
[[nodiscard]] auto
tuneFrameExtent(alpaka::onHost::concepts::FrameSpec auto const &, auto values,
                concepts::MdPolicy auto policy) {
  return frameExtent(std::move(values), policy);
}

/** One independent FrameSpec frame-count tuning entry. */
[[nodiscard]] auto
tuneNumFrames(alpaka::onHost::concepts::FrameSpec auto const &, auto values) {
  return numFrames(std::move(values));
}

/** Select how an explicit multidimensional frame-count list is combined. */
[[nodiscard]] auto
tuneNumFrames(alpaka::onHost::concepts::FrameSpec auto const &, auto values,
              concepts::MdPolicy auto policy) {
  return numFrames(std::move(values), policy);
}

/** One independent ThreadSpec block-count tuning entry. */
[[nodiscard]] auto
tuneNumBlocks(alpaka::onHost::concepts::ThreadSpec auto const &, auto values) {
  return numBlocks(std::move(values));
}

/** Select how an explicit multidimensional block-count list is combined. */
[[nodiscard]] auto
tuneNumBlocks(alpaka::onHost::concepts::ThreadSpec auto const &, auto values,
              concepts::MdPolicy auto policy) {
  return numBlocks(std::move(values), policy);
}

/** One independent ThreadSpec thread-count tuning entry. */
[[nodiscard]] auto
tuneNumThreads(alpaka::onHost::concepts::ThreadSpec auto const &, auto values) {
  return numThreads(std::move(values));
}

/** Select how an explicit multidimensional thread-count list is combined. */
[[nodiscard]] auto
tuneNumThreads(alpaka::onHost::concepts::ThreadSpec auto const &, auto values,
               concepts::MdPolicy auto policy) {
  return numThreads(std::move(values), policy);
}

/** Keep the logical coverage of correlated FrameSpec candidates unchanged. */
[[nodiscard]] auto
preserveCoverage(alpaka::onHost::concepts::FrameSpec auto const &frameSpec) {
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

/** Reject FrameSpec candidates that exceed the original logical coverage. */
[[nodiscard]] auto doesNotExceedCoverage(
    alpaka::onHost::concepts::FrameSpec auto const &frameSpec) {
  return alpakaTune::restrict(
      numFrames, frameExtent,
      [originalNumFrames = frameSpec.getNumFrames(),
       originalFrameExtents = frameSpec.getFrameExtents()](
          alpaka::concepts::VectorOrScalar auto const &candidateNumFrames,
          alpaka::concepts::VectorOrScalar auto const &candidateFrameExtent) {
        if constexpr (alpaka::isVector_v<ALPAKA_TYPEOF(candidateNumFrames)> &&
                      alpaka::isVector_v<ALPAKA_TYPEOF(candidateFrameExtent)>) {
          if constexpr (ALPAKA_TYPEOF(candidateNumFrames)::dim() !=
                        ALPAKA_TYPEOF(candidateFrameExtent)::dim())
            return false;
          for (std::size_t dimension = 0u;
               dimension < ALPAKA_TYPEOF(candidateNumFrames)::dim();
               ++dimension)
            if (!detail::productDoesNotExceed(candidateNumFrames[dimension],
                                              candidateFrameExtent[dimension],
                                              originalNumFrames[dimension],
                                              originalFrameExtents[dimension]))
              return false;
          return true;
        } else
          return detail::productDoesNotExceed(
              candidateNumFrames, candidateFrameExtent, originalNumFrames,
              originalFrameExtents);
      });
}

/** Runtime default frame extents with 32 through 1024 logical elements.
 *
 * Multidimensional extents contain every power-of-two factorization of each
 * element count. Components are nondecreasing, placing the largest component
 * in Alpaka's final and fastest-varying index. The original extent is retained
 * as a fallback for small or otherwise nonstandard launch prototypes.
 */
[[nodiscard]] auto defaultFrameExtentCandidates(
    alpaka::onHost::concepts::FrameSpec auto const &frameSpec) {
  using FrameExtents = ALPAKA_TYPEOF(frameSpec.getFrameExtents());
  using Scalar = ALPAKA_TYPEOF(frameSpec.getFrameExtents()[0u]);
  using RuntimeFrameExtents = alpaka::Vec<Scalar, FrameExtents::dim()>;
  if constexpr (std::numeric_limits<Scalar>::max() < 1024u)
    throw std::overflow_error{
        "The FrameSpec index type cannot represent the default extent 1024."};

  auto values = std::vector<RuntimeFrameExtents>{};
  for (auto const elements : detail::defaultFrameElementCounts) {
    auto exponent = std::size_t{};
    for (auto value = elements; value > 1u; value /= 2u)
      ++exponent;
    auto const exponentExtents =
        alpaka::Vec<std::size_t, RuntimeFrameExtents::dim()>::fill(exponent +
                                                                   1u);
    auto exponentCombinationCount = std::size_t{1u};
    for (std::size_t dimension = 0u;
         dimension < ALPAKA_TYPEOF(exponentExtents)::dim(); ++dimension) {
      if (exponentCombinationCount >
          std::numeric_limits<std::size_t>::max() / exponentExtents[dimension])
        throw std::overflow_error{
            "The default frame-extent candidate space is too large."};
      exponentCombinationCount *= exponentExtents[dimension];
    }
    for (auto linearIndex = std::size_t{0u};
         linearIndex < exponentCombinationCount; ++linearIndex) {
      auto const exponents = alpaka::mapToND(exponentExtents, linearIndex);
      auto exponentSum = std::size_t{};
      auto nondecreasing = true;
      for (std::size_t dimension = 0u;
           dimension < ALPAKA_TYPEOF(exponents)::dim(); ++dimension) {
        exponentSum += exponents[dimension];
        if (dimension > 0u && exponents[dimension - 1u] > exponents[dimension])
          nondecreasing = false;
      }
      if (exponentSum != exponent || !nondecreasing)
        continue;
      values.emplace_back([&exponents](auto dimension) {
        auto const index = static_cast<std::size_t>(dimension);
        return static_cast<Scalar>(std::size_t{1u} << exponents[index]);
      });
    }
  }
  detail::appendUnique(values,
                       RuntimeFrameExtents{frameSpec.getFrameExtents()});
  return alpakaTune::RVals<RuntimeFrameExtents>{std::move(values)};
}

/** Runtime numFrames defaults from one through a caller-provided upper limit.
 *
 * Each component contains repeated halves of its upper limit. Every refinement
 * level inserts the integer midpoint between adjacent values; one level keeps
 * the original default, while three levels approximate eighth-interval spacing
 * between successive halvings. Multidimensional candidates are their Cartesian
 * product, with the final Alpaka index varying fastest.
 */
[[nodiscard]] auto
defaultNumFramesCandidates(alpaka::concepts::Vector auto const &upperLimitValue,
                           std::size_t refinementLevels = 1u) {
  using NumFrames = ALPAKA_TYPEOF(upperLimitValue);
  using Scalar = ALPAKA_TYPEOF(upperLimitValue[0u]);
  using RuntimeNumFrames = alpaka::Vec<Scalar, NumFrames::dim()>;
  auto const upperLimit = RuntimeNumFrames{upperLimitValue};
  auto components = std::array<std::vector<Scalar>, RuntimeNumFrames::dim()>{};
  auto candidateCount = std::size_t{1u};
  for (std::size_t dimension = 0u; dimension < RuntimeNumFrames::dim();
       ++dimension) {
    components[dimension] =
        detail::splitCandidates(upperLimit[dimension], refinementLevels);
    if (candidateCount >
        std::numeric_limits<std::size_t>::max() / components[dimension].size())
      throw std::overflow_error{
          "The default numFrames candidate space is too large."};
    candidateCount *= components[dimension].size();
  }
  auto values = std::vector<RuntimeNumFrames>{};
  values.reserve(candidateCount);
  auto const componentExtents =
      alpaka::Vec<std::size_t, RuntimeNumFrames::dim()>{
          [&components](auto dimension) {
            return components[dimension].size();
          }};
  for (auto linearIndex = std::size_t{0u}; linearIndex < candidateCount;
       ++linearIndex) {
    auto const componentIndices =
        alpaka::mapToND(componentExtents, linearIndex);
    values.emplace_back([&](auto dimension) {
      auto const index = static_cast<std::size_t>(dimension);
      return components[index][componentIndices[index]];
    });
  }
  return alpakaTune::RVals<RuntimeNumFrames>{std::move(values)};
}

/** Restrict recombined runtime-vector values to valid default extents. */
[[nodiscard]] auto defaultFrameExtentShape(
    alpaka::onHost::concepts::FrameSpec auto const &frameSpec) {
  using FrameExtents = ALPAKA_TYPEOF(frameSpec.getFrameExtents());
  using Scalar = ALPAKA_TYPEOF(frameSpec.getFrameExtents()[0u]);
  using RuntimeFrameExtents = alpaka::Vec<Scalar, FrameExtents::dim()>;
  return alpakaTune::restrict(
      frameExtent,
      [original = RuntimeFrameExtents{frameSpec.getFrameExtents()}](
          alpaka::concepts::Vector auto const &candidate) {
        return candidate == original || detail::isDefaultFrameExtent(candidate);
      });
}

/** One generated frameExtent entry with its multidimensional shape relation. */
[[nodiscard]] auto makeDefaultFrameExtentTuning(
    alpaka::onHost::concepts::FrameSpec auto const &frameSpec) {
  return alpakaTune::constrain(
      alpakaTune::TunableBundle{
          tuneFrameExtent(frameSpec, defaultFrameExtentCandidates(frameSpec))},
      defaultFrameExtentShape(frameSpec));
}

/** Keep the logical coverage of correlated ThreadSpec candidates unchanged. */
[[nodiscard]] auto
preserveCoverage(alpaka::onHost::concepts::ThreadSpec auto const &threadSpec) {
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

[[nodiscard]] auto
threadCandidates(alpaka::concepts::Vector auto const &numBlocksValue,
                 alpaka::concepts::Vector auto const &numThreadsValue,
                 bool const allowAlternate) {
  using NumBlocks = ALPAKA_TYPEOF(numBlocksValue);
  using NumThreads = ALPAKA_TYPEOF(numThreadsValue);
  using RuntimeNumBlocks =
      alpaka::Vec<ALPAKA_TYPEOF(numBlocksValue[0u]), NumBlocks::dim()>;
  using RuntimeNumThreads =
      alpaka::Vec<ALPAKA_TYPEOF(numThreadsValue[0u]), NumThreads::dim()>;

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

/** Combine independent FrameSpec entries with one or more relations. */
[[nodiscard]] auto makeFrameSpecTuning(auto frameExtentTuning,
                                       auto numFramesTuning,
                                       auto... relations) {
  static_assert(detail::sameName<ALPAKA_TYPEOF(frameExtentTuning)::name,
                                 detail::frameExtentName>,
                "makeFrameSpecTuning expects a tuneFrameExtent entry first.");
  static_assert(detail::sameName<ALPAKA_TYPEOF(numFramesTuning)::name,
                                 detail::numFramesName>,
                "makeFrameSpecTuning expects a tuneNumFrames entry second.");
  static_assert(sizeof...(relations) > 0u,
                "Independent FrameSpec entries belong directly in a "
                "TunableBundle; makeFrameSpecTuning requires a relation.");
  return alpakaTune::constrain(
      alpakaTune::TunableBundle{std::move(frameExtentTuning),
                                std::move(numFramesTuning)},
      std::move(relations)...);
}

/** Generated FrameSpec defaults that never exceed the original coverage. */
[[nodiscard]] auto
makeFrameSpecTuning(alpaka::onHost::concepts::FrameSpec auto const &frameSpec) {
  return alpakaTune::constrain(
      alpakaTune::TunableBundle{
          makeDefaultFrameExtentTuning(frameSpec),
          tuneNumFrames(frameSpec,
                        defaultNumFramesCandidates(frameSpec.getNumFrames()))},
      doesNotExceedCoverage(frameSpec));
}

/** Combine independent ThreadSpec entries with one or more relations. */
[[nodiscard]] auto makeThreadSpecTuning(auto numBlocksTuning,
                                        auto numThreadsTuning,
                                        auto... relations) {
  static_assert(detail::sameName<ALPAKA_TYPEOF(numBlocksTuning)::name,
                                 detail::numBlocksName>,
                "makeThreadSpecTuning expects a tuneNumBlocks entry first.");
  static_assert(detail::sameName<ALPAKA_TYPEOF(numThreadsTuning)::name,
                                 detail::numThreadsName>,
                "makeThreadSpecTuning expects a tuneNumThreads entry second.");
  static_assert(sizeof...(relations) > 0u,
                "Independent ThreadSpec entries belong directly in a "
                "TunableBundle; makeThreadSpecTuning requires a relation.");
  return alpakaTune::constrain(
      alpakaTune::TunableBundle{std::move(numBlocksTuning),
                                std::move(numThreadsTuning)},
      std::move(relations)...);
}

/** Default coverage-preserving ThreadSpec tuning fragment. */
[[nodiscard]] auto makeThreadSpecTuning(
    alpaka::onHost::concepts::ThreadSpec auto const &threadSpec) {
  auto [blockValues, threadValues] = threadCandidates(
      threadSpec.getNumBlocks(), threadSpec.getNumThreads(),
      !alpaka::isSeqExecutor(ALPAKA_TYPEOF(threadSpec)::getExecutor()));
  return makeThreadSpecTuning(
      tuneNumBlocks(threadSpec, std::move(blockValues)),
      tuneNumThreads(threadSpec, std::move(threadValues)),
      preserveCoverage(threadSpec));
}

/** Explicitly permit changed logical coverage for a kernel that traverses its
 * full problem. */
struct FullTraversal {};
inline constexpr FullTraversal fullTraversal{};

namespace detail {
template <FixedString Name, std::size_t I = 0u, typename Default,
          typename Tuple>
auto selectLaunchOverride(Default fallback, Tuple const &choices) {
  if constexpr (I == std::tuple_size_v<Tuple>)
    return fallback;
  else {
    using Entry = std::remove_cvref_t<std::tuple_element_t<I, Tuple>>;
    if constexpr (sameName<Entry::name, Name>)
      return std::get<I>(choices);
    else
      return selectLaunchOverride<Name, I + 1u>(std::move(fallback), choices);
  }
}
template <FixedString Name, typename Default, typename... Overrides>
auto launchOverride(Default fallback, Overrides const &...overrides) {
  static_assert(
      (std::size_t{} + ... + (sameName<Overrides::name, Name> ? 1u : 0u)) <= 1u,
      "A launch parameter can only be overridden once.");
  return selectLaunchOverride<Name>(std::move(fallback),
                                    std::tie(overrides...));
}

// Each component contributes alternatives without materializing their product.
// Runtime vector dimensions retain the existing independent-component contract.
auto automaticGeometry(alpaka::concepts::Vector auto const &counts,
                       alpaka::concepts::Vector auto const &extent,
                       bool sequential) {
  using Vector =
      alpaka::Vec<ALPAKA_TYPEOF(counts[0u]), ALPAKA_TYPEOF(counts)::dim()>;
  auto extents = std::vector<Vector>{Vector{extent}};
  auto numbers = std::vector<Vector>{Vector{counts}};
  for (std::size_t d{}; d < Vector::dim(); ++d) {
    auto const originalCount = static_cast<std::uintmax_t>(counts[d]);
    auto const originalExtent = static_cast<std::uintmax_t>(extent[d]);
    if (originalExtent == 0u || originalCount == 0u ||
        originalCount >
            std::numeric_limits<std::uintmax_t>::max() / originalExtent)
      throw std::invalid_argument{
          "Automatic launch tuning requires positive representable coverage."};
    auto const coverage = originalCount * originalExtent;
    for (std::uintmax_t value{1u}; value <= 1024u; value *= 2u) {
      if (sequential && value != 1u)
        continue;
      if (coverage % value != 0u ||
          coverage / value >
              std::numeric_limits<typename Vector::value_type>::max())
        continue;
      auto shape = Vector{extent};
      shape[d] = static_cast<typename Vector::value_type>(value);
      auto number = Vector{counts};
      number[d] = static_cast<typename Vector::value_type>(coverage / value);
      appendUnique(extents, shape);
      appendUnique(numbers, number);
    }
  }
  return std::pair{autoCandidates(RVals<Vector>{std::move(extents)}),
                   autoCandidates(RVals<Vector>{std::move(numbers)})};
}
} // namespace detail

/** Automatic FrameSpec domains preserving each axis's original coverage. */
template <typename... Overrides>
[[nodiscard]] auto
makeAutomaticLaunchTuning(alpaka::onHost::concepts::FrameSpec auto const &spec,
                          Overrides const &...overrides) {
  static_assert(((detail::sameName<Overrides::name, detail::frameExtentName> ||
                  detail::sameName<Overrides::name, detail::numFramesName>) &&
                 ...),
                "FrameSpec overrides must name frameExtent or numFrames.");
  auto [extentValues, countValues] = detail::automaticGeometry(
      spec.getNumFrames(), spec.getFrameExtents(), false);
  return makeFrameSpecTuning(
      detail::launchOverride<detail::frameExtentName>(
          tuneFrameExtent(spec, std::move(extentValues)), overrides...),
      detail::launchOverride<detail::numFramesName>(
          tuneNumFrames(spec, std::move(countValues)), overrides...),
      preserveCoverage(spec));
}

/** Automatic ThreadSpec domains, checked against the selected executor and
 * device at first enqueue. */
template <typename... Overrides>
[[nodiscard]] auto
makeAutomaticLaunchTuning(alpaka::onHost::concepts::ThreadSpec auto const &spec,
                          Overrides const &...overrides) {
  static_assert(((detail::sameName<Overrides::name, detail::numThreadsName> ||
                  detail::sameName<Overrides::name, detail::numBlocksName>) &&
                 ...),
                "ThreadSpec overrides must name numThreads or numBlocks.");
  auto [threadValues, blockValues] = detail::automaticGeometry(
      spec.getNumBlocks(), spec.getNumThreads(),
      alpaka::isSeqExecutor(ALPAKA_TYPEOF(spec)::getExecutor()));
  return makeThreadSpecTuning(
      detail::launchOverride<detail::numBlocksName>(
          tuneNumBlocks(spec, std::move(blockValues)), overrides...),
      detail::launchOverride<detail::numThreadsName>(
          tuneNumThreads(spec, std::move(threadValues)), overrides...),
      preserveCoverage(spec));
}

/** Wider default FrameSpec geometry requires an explicit full-traversal
 * promise. */
[[nodiscard]] auto
makeAutomaticLaunchTuning(alpaka::onHost::concepts::FrameSpec auto const &spec,
                          FullTraversal) {
  return TunableBundle{
      tuneFrameExtent(spec, autoCandidates(defaultFrameExtentCandidates(spec))),
      tuneNumFrames(spec, autoCandidates(
                              defaultNumFramesCandidates(spec.getNumFrames()))),
  };
}

/** Wider physical geometry requires a kernel that traverses its full problem.
 */
[[nodiscard]] auto
makeAutomaticLaunchTuning(alpaka::onHost::concepts::ThreadSpec auto const &spec,
                          FullTraversal) {
  auto logical =
      alpaka::onHost::FrameSpec{spec.getNumBlocks(), spec.getNumThreads(),
                                ALPAKA_TYPEOF(spec)::getExecutor()};
  return TunableBundle{
      tuneNumThreads(spec,
                     autoCandidates(defaultFrameExtentCandidates(logical))),
      tuneNumBlocks(spec, autoCandidates(defaultNumFramesCandidates(
                              spec.getNumBlocks())))};
}

} // namespace alpakaTune
