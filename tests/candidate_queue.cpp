// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#include <alpakaTune/core/peripherals/CandidateQueue.hpp>

#include <cstdlib>
#include <random>
#include <vector>

auto main() -> int {
  std::mt19937_64 random{0u};
  auto queue = alpakaTune::detail::CandidateQueue{2u, 2u, false, random};

  if (!queue.empty() || !queue.insert(10u) || !queue.insert(20u) ||
      !queue.full() || queue.insert(30u))
    return EXIT_FAILURE;

  auto const first = queue.next();
  auto const second = queue.next();
  auto const third = queue.next();
  if (!first || !second || !third || first->candidateIndex != 10u ||
      !first->beginActivation || second->candidateIndex != 10u ||
      second->beginActivation || third->candidateIndex != 20u ||
      !third->beginActivation)
    return EXIT_FAILURE;

  if (!queue.retire(10u) || queue.size() != 1u || !queue.retire(20u) ||
      !queue.empty() || queue.retire(20u))
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
