// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0
#pragma once

#include "alpakaTune/space/SpaceConfig.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace alpakaTune::detail {

/** One stable-ID provider for fixed Cartesian and feedback-generated spaces. */
template <std::size_t Dimensions> class CandidateSpace {
public:
  using Indices = std::array<std::size_t, Dimensions>;

  void initialise(std::span<std::size_t const> sizes, bool generate,
                  bool enumerable, SpaceConfig config, std::uint64_t seed) {
    m_sizes.assign(sizes.begin(), sizes.end());
    m_generate = generate;
    m_enumerable = enumerable;
    m_config = config;
    m_random.seed(seed);
    m_total = 1u;
    for (auto size : m_sizes) {
      if (size == 0u)
        throw std::invalid_argument{"An empty candidate domain."};
      if (m_total && *m_total <= std::numeric_limits<std::size_t>::max() / size)
        *m_total *= size;
      else
        m_total.reset();
    }
    if (!generate && !m_total)
      throw std::overflow_error{"The Cartesian tuning space is too large."};
  }

  void setHints(Indices preferred, bool hasPreferred,
                std::array<bool, Dimensions> categorical) {
    m_preferred = preferred;
    m_hasPreferred = hasPreferred;
    m_categorical = categorical;
  }
  [[nodiscard]] auto size() const -> std::size_t {
    return m_generate ? m_catalog.size() : m_total.value_or(0u);
  }
  [[nodiscard]] auto usesCandidateCatalog() const noexcept -> bool {
    return m_generate;
  }
  [[nodiscard]] auto indices(std::size_t id) const -> Indices {
    if (id >= size())
      throw std::out_of_range{"Unknown candidate ID."};
    return m_generate ? m_catalog.at(id) : cartesianIndices(id);
  }
  [[nodiscard]] auto find(Indices const &values) const
      -> std::optional<std::size_t> {
    if (!m_generate) {
      std::size_t result{};
      for (std::size_t d{}; d < Dimensions; ++d) {
        if (values[d] >= m_sizes[d])
          return std::nullopt;
        result = result * m_sizes[d] + values[d];
      }
      return result;
    }
    auto found = std::find(m_catalog.begin(), m_catalog.end(), values);
    if (found == m_catalog.end())
      return std::nullopt;
    return static_cast<std::size_t>(found - m_catalog.begin());
  }
  [[nodiscard]] auto state() const noexcept -> SpaceState { return m_state; }
  [[nodiscard]] auto canExpand() const noexcept -> bool {
    return m_generate && m_state == SpaceState::active;
  }
  [[nodiscard]] auto domainExhausted() const noexcept -> bool {
    return !m_generate || m_state == SpaceState::exhausted;
  }
  [[nodiscard]] auto info() const -> SpaceInfo {
    return {.registeredCandidateCount = size(),
            .declaredCombinationCount = m_enumerable ? m_total : std::nullopt,
            .distinctMeasuredCandidateCount = m_measured.size(),
            .revision = m_revision,
            .state = m_state,
            .usesCandidateCatalog = m_generate,
            .domainExhausted = domainExhausted()};
  }

  /** Feedback is counted once per candidate, even with adaptive revisits. */
  auto observe(std::size_t id, double score) -> bool {
    if (id >= size())
      throw std::out_of_range{"Feedback for an unknown candidate ID."};
    if (!m_generate) {
      m_measured.insert(id);
      return false;
    }
    if (m_measured.insert(id).second) {
      ++m_sinceRefinement;
      if (!m_best ||
          score < *m_best * (1.0 - m_config.minimumRelativeImprovement)) {
        m_best = score;
        m_sinceImprovement = 0u;
        m_globalSinceImprovement = 0u;
      } else {
        ++m_sinceImprovement;
        if (m_globalCandidates.at(id))
          ++m_globalSinceImprovement;
      }
      if (m_config.plateauPatience && m_globalSinceImprovement > 0u &&
          m_sinceImprovement >= *m_config.plateauPatience)
        m_state = SpaceState::plateau;
    }
    return m_sinceRefinement >= m_config.refinementInterval && canExpand();
  }

  /** Canonicalize dependencies before validating and registering a tuple. */
  template <typename Canonicalize, typename Accept>
  auto expand(Canonicalize canonicalize, Accept accept,
              std::span<std::size_t const> best = {}) -> std::size_t {
    if (!canExpand())
      return 0u;
    auto const previous = size();
    auto const requested = previous == 0u ? m_config.initialCandidates
                                          : m_config.refinementBatchSize;
    auto const target =
        previous + std::min(requested, m_config.maximumCandidates - previous);
    std::size_t attempts{};
    while (size() < target && attempts++ < m_config.maximumGenerationAttempts) {
      Indices proposal{};
      bool const usePreferred =
          previous == 0u && attempts == 1u && m_hasPreferred;
      bool const local = !best.empty() && attempts % 4u != 1u;
      if (usePreferred)
        proposal = m_preferred;
      else if (local) {
        proposal = indices(best[(attempts / 4u) % best.size()]);
        auto const first =
            (attempts / 4u) % std::max<std::size_t>(1u, Dimensions);
        for (std::size_t change{}; change < (attempts % 2u + 1u); ++change) {
          if constexpr (Dimensions > 0u) {
            auto const d = (first + change) % Dimensions;
            auto const radius = std::max<std::size_t>(
                1u, m_sizes[d] >>
                        std::min<std::size_t>(
                            m_revision + 2u,
                            std::numeric_limits<std::size_t>::digits - 1u));
            auto const lower = !m_categorical[d] && proposal[d] > radius
                                   ? proposal[d] - radius
                                   : 0u;
            auto const upper =
                m_categorical[d]
                    ? m_sizes[d] - 1u
                    : proposal[d] +
                          std::min(radius, m_sizes[d] - 1u - proposal[d]);
            proposal[d] = std::uniform_int_distribution<std::size_t>{
                lower, upper}(m_random);
          }
        }
      } else {
        // Enumeration is a separate cursor, so local sampling cannot falsely
        // certify coverage. Small domains are exhausted without coupon tails.
        if (m_enumerable && m_total &&
            *m_total <= m_config.maximumGenerationAttempts) {
          if (m_cursor == *m_total) {
            m_state = SpaceState::exhausted;
            break;
          }
          proposal = cartesianIndices(m_cursor++);
        } else {
          for (std::size_t d{}; d < Dimensions; ++d)
            proposal[d] = std::uniform_int_distribution<std::size_t>{
                0u, m_sizes[d] - 1u}(m_random);
          if (previous == 0u && attempts <= 3u)
            for (std::size_t d{}; d < Dimensions; ++d)
              proposal[d] = attempts == 1u   ? 0u
                            : attempts == 2u ? m_sizes[d] - 1u
                                             : m_sizes[d] / 2u;
        }
      }
      auto const sampleGlobal =
          !usePreferred && !local &&
          !(m_enumerable && m_total &&
            *m_total <= m_config.maximumGenerationAttempts);
      if (!canonicalize(proposal, sampleGlobal) || !accept(proposal) ||
          find(proposal))
        continue;
      m_catalog.push_back(proposal);
      m_globalCandidates.push_back(!local || usePreferred);
    }
    m_sinceRefinement = 0u;
    if (size() != previous)
      ++m_revision;
    if (m_enumerable && m_total && (m_cursor == *m_total || size() == *m_total))
      m_state = SpaceState::exhausted;
    else if (size() >= m_config.maximumCandidates)
      m_state = SpaceState::candidateBudget;
    else if (size() == previous && m_state == SpaceState::active)
      m_state = SpaceState::stalled;
    return size() - previous;
  }

  /** Catalog restoration never regenerates or renumbers a saved candidate. */
  void restore(std::vector<Indices> catalog, std::size_t cursor,
               std::size_t revision, SpaceState state,
               std::vector<bool> global = {}) {
    if (!m_generate)
      throw std::logic_error{
          "Cannot restore a generated catalog into a Cartesian provider."};
    if (catalog.size() > m_config.maximumCandidates)
      throw std::invalid_argument{
          "Saved catalog exceeds the configured candidate budget."};
    auto validated = std::vector<Indices>{};
    for (auto const &row : catalog) {
      for (std::size_t d{}; d < Dimensions; ++d)
        if (row[d] >= m_sizes[d])
          throw std::invalid_argument{"Saved candidate is outside its domain."};
      if (std::find(validated.begin(), validated.end(), row) != validated.end())
        throw std::invalid_argument{"Duplicate saved candidate ID."};
      validated.push_back(row);
    }
    if (m_total && cursor > *m_total)
      throw std::invalid_argument{
          "Saved enumeration cursor is outside its domain."};
    if (global.empty())
      global.assign(validated.size(), true);
    if (global.size() != validated.size())
      throw std::invalid_argument{
          "Saved probe flags do not match the catalog."};
    m_globalCandidates = std::move(global);
    m_catalog = std::move(validated);
    m_cursor = cursor;
    m_revision = revision;
    m_state = state;
    if (state == SpaceState::candidateBudget &&
        size() < m_config.maximumCandidates)
      m_state = SpaceState::active;
    if (state == SpaceState::plateau && !m_config.plateauPatience)
      m_state = SpaceState::active;
  }
  [[nodiscard]] auto catalog() const -> std::vector<Indices> const & {
    return m_catalog;
  }
  [[nodiscard]] auto globalCandidates() const -> std::vector<bool> const & {
    return m_globalCandidates;
  }
  [[nodiscard]] auto cursor() const -> std::size_t { return m_cursor; }
  [[nodiscard]] auto revision() const -> std::size_t { return m_revision; }
  [[nodiscard]] auto randomState() const -> std::string {
    std::ostringstream stream;
    stream << m_random;
    return stream.str();
  }
  void restoreRandomState(std::string const &state) {
    std::istringstream stream{state};
    auto restored = std::mt19937_64{};
    if (!(stream >> restored))
      throw std::invalid_argument{"Invalid saved generator state."};
    m_random = restored;
  }
  void freeze() noexcept {
    if (canExpand())
      m_state = SpaceState::candidateBudget;
  }

private:
  [[nodiscard]] auto cartesianIndices(std::size_t value) const -> Indices {
    Indices result{};
    for (std::size_t p = Dimensions; p > 0u; --p) {
      auto const d = p - 1u;
      result[d] = value % m_sizes[d];
      value /= m_sizes[d];
    }
    return result;
  }
  Indices m_preferred{};
  bool m_hasPreferred{};
  std::array<bool, Dimensions> m_categorical{};
  std::vector<std::size_t> m_sizes;
  std::optional<std::size_t> m_total;
  std::vector<Indices> m_catalog;
  std::vector<bool> m_globalCandidates;
  std::set<std::size_t> m_measured;
  SpaceConfig m_config;
  std::mt19937_64 m_random;
  bool m_generate{};
  bool m_enumerable{true};
  SpaceState m_state{SpaceState::active};
  std::size_t m_cursor{}, m_revision{}, m_sinceRefinement{},
      m_sinceImprovement{}, m_globalSinceImprovement{};
  std::optional<double> m_best;
};
} // namespace alpakaTune::detail
