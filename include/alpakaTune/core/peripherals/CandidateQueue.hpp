// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <cstddef>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

namespace alpakaTune::detail {

/**
 * Bounded, interleaving scheduler for active runtime-tuning candidates.
 *
 * The queue owns scheduling only: it interleaves records and bounds their
 * consecutive launches. The tuner owns the record lifecycle and explicitly
 * retires a candidate once its statistical criterion is met.
 */
class CandidateQueue {
public:
  struct Selection {
    std::size_t candidateIndex;
    /** True only when this candidate becomes active after another record. */
    bool beginActivation;
  };

  CandidateQueue(std::size_t activeWindow, std::size_t maxConsecutiveRuns,
                 bool randomOrder, std::mt19937_64 &random)
      : m_slots(activeWindow), m_maxConsecutiveRuns(maxConsecutiveRuns),
        m_randomOrder(randomOrder), m_random(random) {
    if (activeWindow == 0u)
      throw std::invalid_argument{
          "The noise-cancellation window must contain at least one candidate."};
    if (maxConsecutiveRuns == 0u)
      throw std::invalid_argument{
          "The maximum consecutive candidate runs must be greater than zero."};
  }

  [[nodiscard]] bool empty() const noexcept { return m_activeCount == 0u; }
  [[nodiscard]] bool full() const noexcept {
    return m_activeCount == m_slots.size();
  }
  [[nodiscard]] std::size_t size() const noexcept { return m_activeCount; }

  /** Add one configuration record to the active measurement window. */
  bool insert(std::size_t candidateIndex) {
    for (auto &slot : m_slots) {
      if (!slot.has_value()) {
        slot = Slot{candidateIndex};
        ++m_activeCount;
        return true;
      }
    }
    return false;
  }

  /** Retire a configuration whose local measurement lifecycle has finished. */
  bool retire(std::size_t candidateIndex) {
    for (std::size_t index = 0u; index < m_slots.size(); ++index) {
      auto &slot = m_slots.at(index);
      if (!slot || slot->candidateIndex != candidateIndex)
        continue;
      slot.reset();
      --m_activeCount;
      if (m_lastSlot == index) {
        m_lastSlot.reset();
        m_consecutiveRuns = 0u;
      }
      return true;
    }
    return false;
  }

  /** Select an active candidate without consuming its statistical budget. */
  [[nodiscard]] std::optional<Selection> next() {
    if (empty())
      return std::nullopt;

    if (mayReuseLastSlot()) {
      ++m_consecutiveRuns;
      return Selection{m_slots.at(*m_lastSlot)->candidateIndex, false};
    }

    auto const previous = m_lastSlot;
    auto const selected = selectNewSlot();
    m_lastSlot = selected;
    m_consecutiveRuns = 1u;
    m_nextSlot = (selected + 1u) % m_slots.size();
    return Selection{m_slots.at(selected)->candidateIndex,
                     !previous || *previous != selected};
  }

private:
  struct Slot {
    std::size_t candidateIndex;
  };

  [[nodiscard]] bool mayReuseLastSlot() const {
    return m_lastSlot.has_value() && m_slots.at(*m_lastSlot).has_value() &&
           m_consecutiveRuns < m_maxConsecutiveRuns;
  }

  [[nodiscard]] std::size_t selectNewSlot() {
    auto selectableSlots = std::vector<std::size_t>{};
    selectableSlots.reserve(m_activeCount);
    for (std::size_t index = 0u; index < m_slots.size(); ++index) {
      if (!m_slots[index].has_value())
        continue;
      if (m_activeCount > 1u && m_lastSlot == index)
        continue;
      selectableSlots.push_back(index);
    }
    if (selectableSlots.empty())
      throw std::logic_error{"The candidate queue has no selectable slot."};

    if (m_randomOrder) {
      std::uniform_int_distribution<std::size_t> distribution{
          0u, selectableSlots.size() - 1u};
      return selectableSlots.at(distribution(m_random));
    }
    for (std::size_t offset = 0u; offset < m_slots.size(); ++offset) {
      auto const candidate = (m_nextSlot + offset) % m_slots.size();
      for (auto const selectable : selectableSlots) {
        if (selectable == candidate)
          return candidate;
      }
    }
    throw std::logic_error{"The candidate queue lost its active slot."};
  }

  std::vector<std::optional<Slot>> m_slots;
  std::size_t m_maxConsecutiveRuns;
  bool m_randomOrder;
  std::mt19937_64 &m_random;
  std::size_t m_activeCount{};
  std::optional<std::size_t> m_lastSlot;
  std::size_t m_consecutiveRuns{};
  std::size_t m_nextSlot{};
};

} // namespace alpakaTune::detail
