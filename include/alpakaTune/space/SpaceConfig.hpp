// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0
#pragma once

#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>

namespace alpakaTune {

/** Budgets for candidate construction; these do not disable adaptation. */
struct SpaceConfig {
  std::size_t initialCandidates{32u};
  std::size_t refinementBatchSize{16u};
  std::size_t refinementInterval{16u};
  std::size_t maximumCandidates{4096u};
  std::size_t maximumGenerationAttempts{2048u};
  std::optional<std::size_t> plateauPatience;
  double minimumRelativeImprovement{0.01};

  void validate() const {
    if (initialCandidates == 0u || refinementBatchSize == 0u ||
        refinementInterval == 0u || maximumCandidates == 0u ||
        maximumGenerationAttempts == 0u ||
        (plateauPatience && *plateauPatience == 0u) ||
        !std::isfinite(minimumRelativeImprovement) ||
        minimumRelativeImprovement < 0.0 || minimumRelativeImprovement >= 1.0)
      throw std::invalid_argument{"Invalid candidate-space budget."};
  }
};

enum class SpaceState { active, exhausted, candidateBudget, plateau, stalled };

[[nodiscard]] constexpr auto spaceStateName(SpaceState state) -> char const * {
  switch (state) {
  case SpaceState::active:
    return "active";
  case SpaceState::exhausted:
    return "exhausted";
  case SpaceState::candidateBudget:
    return "candidate_budget";
  case SpaceState::plateau:
    return "plateau";
  case SpaceState::stalled:
    return "stalled";
  }
  return "active";
}

struct SpaceInfo {
  std::size_t registeredCandidateCount{};
  std::optional<std::size_t> declaredCombinationCount;
  std::size_t distinctMeasuredCandidateCount{};
  std::size_t revision{};
  SpaceState state{SpaceState::active};
  bool usesCandidateCatalog{};
  bool domainExhausted{};
};
} // namespace alpakaTune
