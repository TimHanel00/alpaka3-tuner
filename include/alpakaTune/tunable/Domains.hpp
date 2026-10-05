// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0
#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace alpakaTune {
namespace hint {
struct Linear {};
struct Logarithmic {};
struct Categorical {};
inline constexpr Linear linear{};
inline constexpr Logarithmic logarithmic{};
inline constexpr Categorical categorical{};
struct Alignment {
  std::uint64_t value;
};
[[nodiscard]] constexpr auto alignment(std::uint64_t value) -> Alignment {
  return {value};
}
template <typename T> struct Preferred {
  T value;
};
template <typename T>
[[nodiscard]] constexpr auto preferred(T value) -> Preferred<T> {
  return {value};
}
} // namespace hint

namespace domain {
/** A bounded runtime numeric domain, sampled without materializing its range.
 */
template <typename T>
  requires std::integral<T> || std::floating_point<T>
class Interval {
public:
  using value_type = T;
  static constexpr bool runtimeDomain = true;
  static constexpr bool enumerable = std::integral<T>;
  Interval(T minimum, T maximum, T step = T{1})
      : m_minimum(minimum), m_maximum(maximum), m_step(step) {
    if (!std::isfinite(static_cast<long double>(minimum)) ||
        !std::isfinite(static_cast<long double>(maximum)) ||
        !std::isfinite(static_cast<long double>(step)) || maximum < minimum ||
        step <= T{0})
      throw std::invalid_argument{"A numeric domain requires finite bounds, "
                                  "min <= max, and a positive step."};
    if constexpr (std::integral<T>) {
      using U = std::make_unsigned_t<T>;
      auto const distance = static_cast<U>(maximum) - static_cast<U>(minimum);
      auto const quotient = static_cast<U>(distance / static_cast<U>(step));
      if (quotient >= std::numeric_limits<std::size_t>::max())
        throw std::overflow_error{
            "The domain has too many addressable values."};
      m_size = static_cast<std::size_t>(quotient) + 1u;
    } else
      m_size = minimum == maximum ? 1u : (1u << 24u) + 1u;
  }
  [[nodiscard]] auto size() const noexcept -> std::size_t { return m_size; }
  [[nodiscard]] auto minimum() const noexcept -> T { return m_minimum; }
  [[nodiscard]] auto maximum() const noexcept -> T { return m_maximum; }
  [[nodiscard]] auto step() const noexcept -> T { return m_step; }
  [[nodiscard]] auto at(std::size_t index) const -> T {
    if (index >= size())
      throw std::out_of_range{"Numeric domain index out of range."};
    if constexpr (std::integral<T>) {
      using U = std::make_unsigned_t<T>;
      return static_cast<T>(static_cast<U>(m_minimum) +
                            static_cast<U>(index) * static_cast<U>(m_step));
    } else {
      if (size() == 1u)
        return m_minimum;
      auto const ratio = static_cast<long double>(index) /
                         static_cast<long double>(size() - 1u);
      return static_cast<T>((1.0L - ratio) * m_minimum + ratio * m_maximum);
    }
  }
  [[nodiscard]] auto indexOf(T value) const -> std::optional<std::size_t> {
    if (value < m_minimum || value > m_maximum)
      return std::nullopt;
    if constexpr (std::integral<T>) {
      using U = std::make_unsigned_t<T>;
      auto const offset = static_cast<U>(value) - static_cast<U>(m_minimum);
      if (offset % static_cast<U>(m_step) != 0u)
        return std::nullopt;
      return static_cast<std::size_t>(offset / static_cast<U>(m_step));
    } else {
      if (!std::isfinite(value))
        return std::nullopt;
      if (size() == 1u)
        return 0u;
      auto const ratio = (static_cast<long double>(value) - m_minimum) /
                         (static_cast<long double>(m_maximum) - m_minimum);
      return static_cast<std::size_t>(std::llround(ratio * (size() - 1u)));
    }
  }

private:
  T m_minimum, m_maximum, m_step;
  std::size_t m_size{};
};
template <typename T>
[[nodiscard]] auto interval(T minimum, T maximum, T step = T{1})
    -> Interval<T> {
  return {minimum, maximum, step};
}

/** A child domain constrained by values of named parent parameters. */
template <typename Domain, typename Generator, typename... Names>
struct Dependent : Domain {
  static_assert(
      requires(Domain const &domain) {
        typename Domain::value_type;
        domain.at(0u);
        domain.size();
      }, "A dependent child domain must supply runtime values; compiled "
         "parameters can be parents.");
  static constexpr bool runtimeDomain = true;
  using parent_names = std::tuple<Names...>;
  Generator generator;
  explicit Dependent(Domain values, Generator function)
      : Domain(std::move(values)), generator(std::move(function)) {}
};
template <typename Domain, typename Name, typename Generator>
[[nodiscard]] auto dependent(Domain values, Name, Generator generator) {
  return Dependent<Domain, Generator, Name>{std::move(values),
                                            std::move(generator)};
}
template <typename Domain, typename... Names, typename Generator>
[[nodiscard]] auto dependent(Domain values, std::tuple<Names...>,
                             Generator generator) {
  return Dependent<Domain, Generator, Names...>{std::move(values),
                                                std::move(generator)};
}
} // namespace domain

/** Opt a domain into hardware- and feedback-informed candidate generation. */
template <typename Domain> class AutoCandidates : public Domain {
public:
  using domain_type = Domain;
  static constexpr bool automaticCandidates = true;
  explicit AutoCandidates(Domain values) : Domain(std::move(values)) {}
  void setHint(hint::Linear) { m_logarithmic = false; }
  void setHint(hint::Logarithmic) {
    if constexpr (requires { this->minimum(); })
      if (this->minimum() <= 0)
        throw std::invalid_argument{
            "Logarithmic domains require positive values."};
    m_logarithmic = true;
  }
  void setHint(hint::Categorical) { m_categorical = true; }
  void setHint(hint::Alignment value) {
    if (value.value == 0u)
      throw std::invalid_argument{"Alignment must be positive."};
    m_alignment = value.value;
  }
  template <typename T> void setHint(hint::Preferred<T> value) {
    if constexpr (requires { this->indexOf(value.value); }) {
      m_preferred = this->indexOf(value.value);
      if (!m_preferred)
        throw std::invalid_argument{"Preferred value is outside its domain."};
    } else if constexpr (requires { this->values(); }) {
      auto const &values = this->values();
      for (std::size_t i{}; i < values.size(); ++i)
        if (values[i] == value.value)
          m_preferred = i;
      if (!m_preferred)
        throw std::invalid_argument{"Preferred value is not an alternative."};
    } else if constexpr (requires { typename Domain::values; }) {
      [&]<std::size_t... I>(std::index_sequence<I...>) {
        (
            [&] {
              using Value = std::tuple_element_t<I, typename Domain::values>;
              if constexpr (requires { Value::value == value.value; })
                if (Value::value == value.value)
                  m_preferred = I;
            }(),
            ...);
      }(std::make_index_sequence<Domain::size>{});
      if (!m_preferred)
        throw std::invalid_argument{
            "Preferred value is not a compiled alternative."};
    } else
      throw std::invalid_argument{
          "Preferred hint requires numeric alternatives."};
  }
  [[nodiscard]] auto preferredIndex() const -> std::optional<std::size_t> {
    return m_preferred;
  }
  [[nodiscard]] auto logarithmic() const -> bool { return m_logarithmic; }
  [[nodiscard]] auto categorical() const -> bool { return m_categorical; }
  [[nodiscard]] auto alignment() const -> std::uint64_t { return m_alignment; }
  [[nodiscard]] auto project(std::size_t index, bool sampleGlobal = false) const
      -> std::optional<std::size_t> {
    if constexpr (requires {
                    this->minimum();
                    this->indexOf(this->at(index));
                  }) {
      using T = typename Domain::value_type;
      auto value = this->at(index);
      if (sampleGlobal && m_logarithmic && this->size() > 1u) {
        auto const fraction =
            static_cast<long double>(index) / (this->size() - 1u);
        auto const sampled = std::exp(
            std::log(static_cast<long double>(this->minimum())) +
            fraction * (std::log(static_cast<long double>(this->maximum())) -
                        std::log(static_cast<long double>(this->minimum()))));
        if constexpr (std::integral<T>) {
          auto const position =
              std::round((sampled - this->minimum()) / this->step());
          index = std::min(this->size() - 1u,
                           static_cast<std::size_t>(std::max(0.0L, position)));
        } else
          index = *this->indexOf(static_cast<T>(sampled));
        value = this->at(index);
      }
      if constexpr (std::integral<T>) {
        if (m_alignment > 1u) {
          auto aligned =
              std::floor(static_cast<long double>(value) / m_alignment) *
              m_alignment;
          if (aligned < this->minimum())
            aligned += m_alignment;
          if (aligned > this->maximum())
            return std::nullopt;
          return this->indexOf(static_cast<T>(aligned));
        }
      } else
        return this->indexOf(value);
    }
    return index;
  }

private:
  bool m_logarithmic{}, m_categorical{};
  std::uint64_t m_alignment{1u};
  std::optional<std::size_t> m_preferred;
};
template <typename Domain, typename... Hints>
[[nodiscard]] auto autoCandidates(Domain values, Hints... hints)
    -> AutoCandidates<Domain> {
  auto result = AutoCandidates<Domain>{std::move(values)};
  (result.setHint(hints), ...);
  return result;
}
} // namespace alpakaTune
