// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#include <alpakaTune/tunable/FrameSpecTuning.hpp>

#include <alpaka/alpaka.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
[[nodiscard]] auto hasValue(auto const &values, auto const &expected) -> bool {
  return std::ranges::find(values, expected) != values.end();
}

[[nodiscard]] auto
hasDefaultExtentProperties(alpaka::concepts::Vector auto const &extent)
    -> bool {
  for (std::size_t dimension = 1u; dimension < extent.dim(); ++dimension)
    if (extent[dimension - 1u] > extent[dimension])
      return false;
  auto const elements = static_cast<std::size_t>(extent.product());
  constexpr auto expected =
      std::array<std::size_t, 6u>{32u, 64u, 128u, 256u, 512u, 1024u};
  if (std::ranges::find(expected, elements) == expected.end())
    return false;

  // Alpaka linearizes the final dimension fastest. The generated extent puts
  // its largest component there, and mapToND advances it first.
  auto const mapped = alpaka::mapToND(extent, ALPAKA_TYPEOF(extent[0u]){1u});
  return mapped[extent.dim() - 1u] == ALPAKA_TYPEOF(extent[0u]){1u};
}
} // namespace

int main() {
  auto const executor = alpaka::exec::cpuSerial;

  using Vector1 = alpaka::Vec<std::size_t, 1u>;
  auto const frame1 =
      alpaka::onHost::FrameSpec{Vector1{1024u}, Vector1{16u}, executor};
  auto const extents1 =
      alpakaTune::defaultFrameExtentCandidates(frame1).values();
  auto const expected1 = std::vector<Vector1>{
      Vector1{32u},  Vector1{64u},   Vector1{128u}, Vector1{256u},
      Vector1{512u}, Vector1{1024u}, Vector1{16u}};
  if (extents1 != expected1)
    return EXIT_FAILURE;

  using Vector2 = alpaka::Vec<std::size_t, 2u>;
  auto const originalExtent2 = Vector2{3u, 5u};
  auto const frame2 = alpaka::onHost::FrameSpec{Vector2::fill(1024u),
                                                originalExtent2, executor};
  auto const extents2 =
      alpakaTune::defaultFrameExtentCandidates(frame2).values();
  if (extents2.size() != 28u || extents2[0u] != Vector2{1u, 32u} ||
      extents2[1u] != Vector2{2u, 16u} || extents2[2u] != Vector2{4u, 8u} ||
      extents2.back() != originalExtent2)
    return EXIT_FAILURE;
  for (auto const &extent : extents2)
    if (extent != originalExtent2 && !hasDefaultExtentProperties(extent))
      return EXIT_FAILURE;

  using Vector3 = alpaka::Vec<std::size_t, 3u>;
  auto const originalExtent3 = Vector3{3u, 5u, 7u};
  auto const frame3 = alpaka::onHost::FrameSpec{Vector3::fill(1024u),
                                                originalExtent3, executor};
  auto const extents3 =
      alpakaTune::defaultFrameExtentCandidates(frame3).values();
  if (!hasValue(extents3, Vector3{1u, 1u, 32u}) ||
      !hasValue(extents3, Vector3{2u, 4u, 8u}) ||
      !hasValue(extents3, Vector3{8u, 8u, 16u}) ||
      extents3.back() != originalExtent3)
    return EXIT_FAILURE;
  for (auto const &extent : extents3)
    if (extent != originalExtent3 && !hasDefaultExtentProperties(extent))
      return EXIT_FAILURE;

  auto const numFrames1 =
      alpakaTune::defaultNumFramesCandidates(Vector1{1024u}).values();
  auto const expectedNumFrames1 = std::vector<Vector1>{
      Vector1{1u},   Vector1{2u},   Vector1{3u},   Vector1{4u},
      Vector1{6u},   Vector1{8u},   Vector1{12u},  Vector1{16u},
      Vector1{24u},  Vector1{32u},  Vector1{48u},  Vector1{64u},
      Vector1{96u},  Vector1{128u}, Vector1{192u}, Vector1{256u},
      Vector1{384u}, Vector1{512u}, Vector1{768u}, Vector1{1024u}};
  if (numFrames1 != expectedNumFrames1)
    return EXIT_FAILURE;

  auto const refinedNumFrames =
      alpakaTune::defaultNumFramesCandidates(Vector1{16u}, 2u).values();
  auto const expectedRefinedNumFrames = std::vector<Vector1>{
      Vector1{1u},  Vector1{2u},  Vector1{3u},  Vector1{4u},
      Vector1{5u},  Vector1{6u},  Vector1{7u},  Vector1{8u},
      Vector1{10u}, Vector1{12u}, Vector1{14u}, Vector1{16u}};
  if (refinedNumFrames != expectedRefinedNumFrames)
    return EXIT_FAILURE;

  auto const numFrames2 =
      alpakaTune::defaultNumFramesCandidates(Vector2{4u, 2u}).values();
  auto const expectedNumFrames2 = std::vector<Vector2>{
      Vector2{1u, 1u}, Vector2{1u, 2u}, Vector2{2u, 1u}, Vector2{2u, 2u},
      Vector2{3u, 1u}, Vector2{3u, 2u}, Vector2{4u, 1u}, Vector2{4u, 2u}};
  if (numFrames2 != expectedNumFrames2)
    return EXIT_FAILURE;

  auto zeroUpperLimitRejected = false;
  try {
    static_cast<void>(alpakaTune::defaultNumFramesCandidates(Vector1{0u}));
  } catch (std::invalid_argument const &) {
    zeroUpperLimitRejected = true;
  }
  if (!zeroUpperLimitRejected)
    return EXIT_FAILURE;

  auto const coverageFrame =
      alpaka::onHost::FrameSpec{Vector2{4u, 8u}, Vector2{16u, 8u}, executor};
  auto const coverage = alpakaTune::doesNotExceedCoverage(coverageFrame);
  if (!coverage.accepts(Vector2{2u, 4u}, Vector2{32u, 16u}) ||
      !coverage.accepts(Vector2{1u, 1u}, Vector2{32u, 32u}) ||
      coverage.accepts(Vector2{3u, 4u}, Vector2{32u, 16u}) ||
      coverage.accepts(Vector2{0u, 1u}, Vector2{32u, 16u}))
    return EXIT_FAILURE;

  auto const extentShape = alpakaTune::defaultFrameExtentShape(frame2);
  if (!extentShape.accepts(originalExtent2) ||
      !extentShape.accepts(Vector2{2u, 16u}) ||
      extentShape.accepts(Vector2{16u, 2u}) ||
      extentShape.accepts(Vector2{4u, 4u}))
    return EXIT_FAILURE;

  using LimitVector = alpaka::Vec<std::uint64_t, 1u>;
  auto const maximum = std::numeric_limits<std::uint64_t>::max();
  auto const overflowingFrame = alpaka::onHost::FrameSpec{
      LimitVector{maximum}, LimitVector{2u}, executor};
  auto overflowDetected = false;
  try {
    static_cast<void>(alpakaTune::doesNotExceedCoverage(overflowingFrame)
                          .accepts(LimitVector{1u}, LimitVector{1u}));
  } catch (std::overflow_error const &) {
    overflowDetected = true;
  }
  return overflowDetected ? EXIT_SUCCESS : EXIT_FAILURE;
}
