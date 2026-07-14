// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace alpakaTune {

/** A structural string used as a C++20 non-type template parameter. */
template <std::size_t Size> struct FixedString {
  char value[Size];

  constexpr FixedString(char const (&text)[Size]) : value{} {
    for (std::size_t index = 0u; index < Size; ++index)
      value[index] = text[index];
  }

  [[nodiscard]] constexpr auto view() const -> std::string_view {
    return {value, Size - 1u};
  }

  constexpr auto operator<=>(FixedString const &) const = default;
};

template <FixedString Name> struct TunableName {
  static constexpr auto value = Name;
  [[nodiscard]] constexpr auto view() const -> std::string_view {
    return Name.view();
  }
};

#define ALPAKA_TUNE_NAME(text)                                                   \
  ::alpakaTune::TunableName<::alpakaTune::FixedString{text}>{}

/** Runtime candidate values. Integral constructors describe inclusive ranges. */
template <typename T> class RVals {
public:
  using value_type = T;

  explicit RVals(std::vector<T> values) : m_values(std::move(values)) {
    validate();
  }

  RVals(std::initializer_list<T> values) : RVals(std::vector<T>{values}) {}

  explicit RVals(T maximum)
    requires std::integral<T>
      : RVals(T{0}, maximum, T{1}) {}

  RVals(T minimum, T maximum)
    requires std::integral<T>
      : RVals(minimum, maximum, T{1}) {}

  RVals(T minimum, T maximum, T step)
    requires std::integral<T>
  {
    if (step <= T{0} || maximum < minimum)
      throw std::invalid_argument{"An RVals range requires min <= max and a positive step."};
    for (auto value = minimum;;) {
      m_values.push_back(value);
      if (value > maximum - step)
        break;
      value = static_cast<T>(value + step);
    }
    validate();
  }

  template <typename... Values>
  [[nodiscard]] static auto list(Values &&...values) -> RVals {
    return RVals{std::vector<T>{static_cast<T>(std::forward<Values>(values))...}};
  }

  [[nodiscard]] auto values() const noexcept -> std::vector<T> const & {
    return m_values;
  }

  [[nodiscard]] auto size() const noexcept -> std::size_t { return m_values.size(); }

private:
  void validate() const {
    if (m_values.empty())
      throw std::invalid_argument{"RVals requires at least one candidate value."};
  }

  std::vector<T> m_values;
};

template <typename T> RVals(std::vector<T>) -> RVals<T>;
template <typename T> RVals(std::initializer_list<T>) -> RVals<T>;
template <std::integral T> RVals(T) -> RVals<T>;
template <std::integral T> RVals(T, T) -> RVals<T>;
template <std::integral T> RVals(T, T, T) -> RVals<T>;

/** Compile-time candidate values. */
template <auto... Values> struct CVals {
  static_assert(sizeof...(Values) > 0u, "CVals requires at least one candidate value.");
  static constexpr std::size_t size = sizeof...(Values);
  using values = std::tuple<std::integral_constant<decltype(Values), Values>...>;
};

namespace detail {

template <auto Current, auto Maximum, auto Step, bool Complete,
          auto... Values>
struct MakeCValsRangeImpl;

template <auto Current, auto Maximum, auto Step, auto... Values>
struct MakeCValsRangeImpl<Current, Maximum, Step, false, Values...> {
  using next_type = decltype(Current + Step);
  static constexpr auto next = static_cast<next_type>(Current + Step);
  using type = typename MakeCValsRangeImpl<next, Maximum, Step,
                                            (next > Maximum), Values...,
                                            Current>::type;
};

template <auto Current, auto Maximum, auto Step, auto... Values>
struct MakeCValsRangeImpl<Current, Maximum, Step, true, Values...> {
  using type = CVals<Values...>;
};

template <auto Minimum, auto Maximum, auto Step, auto... Values>
struct MakeCValsRange {
  static_assert(Step > 0, "CValsRange requires a positive step.");
  static_assert(Minimum <= Maximum, "CValsRange requires min <= max.");
  using type = typename MakeCValsRangeImpl<Minimum, Maximum, Step,
                                            (Minimum > Maximum),
                                            Values...>::type;
};

template <typename T> struct IsRVals : std::false_type {};
template <typename T> struct IsRVals<RVals<T>> : std::true_type {};
template <typename T>
inline constexpr bool isRVals = IsRVals<std::remove_cvref_t<T>>::value;

template <typename T> struct IsCVals : std::false_type {};
template <auto... Values> struct IsCVals<CVals<Values...>> : std::true_type {};
template <typename T>
inline constexpr bool isCVals = IsCVals<std::remove_cvref_t<T>>::value;

} // namespace detail

template <auto Maximum>
using CValsTo = typename detail::MakeCValsRange<0, Maximum, 1>::type;

template <auto Minimum, auto Maximum, auto Step = 1>
using CValsRange = typename detail::MakeCValsRange<Minimum, Maximum, Step>::type;

template <FixedString Name, typename Values> struct NamedTunable {
  static constexpr auto name = Name;
  using values_type = Values;
  Values values;

  [[nodiscard]] static constexpr auto nameView() -> std::string_view {
    return Name.view();
  }
};

template <FixedString Name, typename Values>
[[nodiscard]] constexpr auto named(TunableName<Name>, Values values)
    -> NamedTunable<Name, Values> {
  static_assert(detail::isRVals<Values> || detail::isCVals<Values>,
                "A tunable needs RVals or CVals.");
  return {std::move(values)};
}

/** Heterogeneous, statically typed set of named tuning dimensions. */
template <typename... Entries> class Tunables {
public:
  explicit constexpr Tunables(Entries... entries)
      : m_entries(std::move(entries)...) {
    static_assert((requires { Entries::name; } && ...),
                  "Tunables entries must be created with alpakaTune::named.");
  }

  [[nodiscard]] constexpr auto entries() const noexcept -> std::tuple<Entries...> const & {
    return m_entries;
  }

  static constexpr auto size = sizeof...(Entries);

private:
  std::tuple<Entries...> m_entries;
};

template <typename... Entries> Tunables(Entries...) -> Tunables<Entries...>;

/** Placeholder argument in a prototype alpaka::KernelBundle. */
template <FixedString Name> struct MarkedTunable {
  static constexpr auto name = Name;
};

template <FixedString Name>
[[nodiscard]] constexpr auto markTunable(TunableName<Name>) -> MarkedTunable<Name> {
  return {};
}

namespace detail {

template <typename T> struct IsMarkedTunable : std::false_type {};
template <FixedString Name>
struct IsMarkedTunable<MarkedTunable<Name>> : std::true_type {
  static constexpr auto name = Name;
};
template <typename T>
inline constexpr bool isMarkedTunable = IsMarkedTunable<std::remove_cvref_t<T>>::value;

template <FixedString Left, FixedString Right>
inline constexpr bool sameName = Left.view() == Right.view();

template <FixedString Name, typename... Entries> struct HasEntry;
template <FixedString Name> struct HasEntry<Name> : std::false_type {};
template <FixedString Name, typename Entry, typename... Remaining>
struct HasEntry<Name, Entry, Remaining...>
    : std::bool_constant<sameName<Name, Entry::name> ||
                         HasEntry<Name, Remaining...>::value> {};
template <FixedString Name, typename... Entries>
inline constexpr bool hasEntry = HasEntry<Name, Entries...>::value;

template <FixedString Name, std::size_t Index, typename... Entries>
struct FindEntryIndexImpl;

template <FixedString Name, std::size_t Index, typename Entry,
          typename... Remaining>
struct FindEntryIndexImpl<Name, Index, Entry, Remaining...>
    : std::conditional_t<sameName<Name, Entry::name>,
                         std::integral_constant<std::size_t, Index>,
                         FindEntryIndexImpl<Name, Index + 1u, Remaining...>> {};

template <FixedString Name, std::size_t Index>
struct FindEntryIndexImpl<Name, Index> {
  static_assert(Name.view().empty(), "A KernelBundle marker refers to an unknown tunable.");
};

template <FixedString Name, typename... Entries>
inline constexpr std::size_t findEntryIndex =
    FindEntryIndexImpl<Name, 0u, Entries...>::value;

template <FixedString Name, typename... Entries>
using findEntry = std::tuple_element_t<findEntryIndex<Name, Entries...>,
                                       std::tuple<Entries...>>;

template <FixedString Name, typename... Entries>
constexpr auto tupleEntry(std::tuple<Entries...> const &entries)
    -> findEntry<Name, Entries...> const & {
  return std::get<findEntryIndex<Name, Entries...>>(entries);
}

template <typename T> struct TunablesTraits;

template <typename... Entries> struct TunablesTraits<Tunables<Entries...>> {
  using entries_type = std::tuple<Entries...>;
  static constexpr std::size_t size = sizeof...(Entries);

  template <FixedString Name>
  static constexpr bool has = hasEntry<Name, Entries...>;

  template <FixedString Name>
  using entry = findEntry<Name, Entries...>;

  template <FixedString Name>
  static constexpr std::size_t index = findEntryIndex<Name, Entries...>;
};

} // namespace detail

} // namespace alpakaTune
