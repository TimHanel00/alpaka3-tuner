// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/CVec.hpp>
#include <alpaka/Vec.hpp>

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

template <FixedString Name, typename Values> struct Tunable;

template <FixedString Name> struct TunableName {
  static constexpr auto value = Name;
  static constexpr auto name = Name;
  static constexpr bool tunableMarker = true;
  [[nodiscard]] constexpr auto view() const -> std::string_view {
    return Name.view();
  }

  template <typename Values>
  [[nodiscard]] constexpr auto operator()(Values values) const
      -> Tunable<Name, std::remove_cvref_t<Values>>;
};

#define ALPAKA_TUNE_TUNABLE(text)                                              \
  ::alpakaTune::TunableName<::alpakaTune::FixedString{text}> {}

// Compatibility spelling for the first frontend draft.
#define ALPAKA_TUNE_NAME(text) ALPAKA_TUNE_TUNABLE(text)

/** Runtime candidate values. Integral constructors describe inclusive ranges.
 */
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
      throw std::invalid_argument{
          "An RVals range requires min <= max and a positive step."};
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
    return RVals{
        std::vector<T>{static_cast<T>(std::forward<Values>(values))...}};
  }

  [[nodiscard]] auto values() const noexcept -> std::vector<T> const & {
    return m_values;
  }

  [[nodiscard]] auto size() const noexcept -> std::size_t {
    return m_values.size();
  }

private:
  void validate() const {
    if (m_values.empty())
      throw std::invalid_argument{
          "RVals requires at least one candidate value."};
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
  static_assert(sizeof...(Values) > 0u,
                "CVals requires at least one candidate value.");
  static constexpr std::size_t size = sizeof...(Values);
  using values =
      std::tuple<std::integral_constant<decltype(Values), Values>...>;
};

/** Compile-time candidate types, for example differently sized alpaka::CVecs.
 */
template <typename... Values> struct CTypes {
  static_assert(sizeof...(Values) > 0u,
                "CTypes requires at least one candidate type.");
  static constexpr std::size_t size = sizeof...(Values);
  using values = std::tuple<Values...>;
};

namespace detail {

template <auto Current, auto Maximum, auto Step, bool Complete, auto... Values>
struct MakeCValsRangeImpl;

template <auto Current, auto Maximum, auto Step, auto... Values>
struct MakeCValsRangeImpl<Current, Maximum, Step, false, Values...> {
  using next_type = decltype(Current + Step);
  static constexpr auto next = static_cast<next_type>(Current + Step);
  using type =
      typename MakeCValsRangeImpl<next, Maximum, Step, (next > Maximum),
                                  Values..., Current>::type;
};

template <auto Current, auto Maximum, auto Step, auto... Values>
struct MakeCValsRangeImpl<Current, Maximum, Step, true, Values...> {
  using type = CVals<Values...>;
};

template <auto Minimum, auto Maximum, auto Step, auto... Values>
struct MakeCValsRange {
  static_assert(Step > 0, "CValsRange requires a positive step.");
  static_assert(Minimum <= Maximum, "CValsRange requires min <= max.");
  using type =
      typename MakeCValsRangeImpl<Minimum, Maximum, Step, (Minimum > Maximum),
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

template <typename T> struct IsCTypes : std::false_type {};
template <typename... Values>
struct IsCTypes<CTypes<Values...>> : std::true_type {};
template <typename T>
inline constexpr bool isCTypes = IsCTypes<std::remove_cvref_t<T>>::value;

template <typename T>
inline constexpr bool isCompileTimeCandidates = isCVals<T> || isCTypes<T>;

template <typename T> struct IntegerSequenceTraits {
  static constexpr bool value = false;
};

template <typename T, T... Values>
struct IntegerSequenceTraits<std::integer_sequence<T, Values...>> {
  static constexpr bool value = true;
  static constexpr std::size_t dimensionCount = sizeof...(Values);
  using value_type = T;

  template <std::size_t Dimension> static consteval auto component() -> T {
    static_assert(Dimension < dimensionCount);
    constexpr std::array values{Values...};
    return values[Dimension];
  }
};

template <typename T>
inline constexpr bool isIntegerSequence =
    IntegerSequenceTraits<std::remove_cvref_t<T>>::value;

template <typename T>
inline constexpr bool isCompileVector =
    alpaka::isCVector_v<std::remove_cvref_t<T>> || isIntegerSequence<T>;

template <typename T>
consteval auto compileVectorDimensionCount() -> std::size_t {
  using Value = std::remove_cvref_t<T>;
  if constexpr (alpaka::isCVector_v<Value>)
    return Value::dim();
  else
    return IntegerSequenceTraits<Value>::dimensionCount;
}

template <typename T, std::size_t Dimension>
consteval auto compileVectorComponent() {
  using Value = std::remove_cvref_t<T>;
  if constexpr (alpaka::isCVector_v<Value>) {
    static_assert(Dimension < Value::dim());
    return Value{}[Dimension];
  } else {
    return IntegerSequenceTraits<Value>::template component<Dimension>();
  }
}

template <typename Values>
struct CandidateDimensionCount : std::integral_constant<std::size_t, 1u> {};

template <typename T, bool = alpaka::isVector_v<T>>
struct RuntimeCandidateDimensionCount
    : std::integral_constant<std::size_t, 1u> {};

template <typename T>
struct RuntimeCandidateDimensionCount<T, true>
    : std::integral_constant<std::size_t, static_cast<std::size_t>(T::dim())> {
};

template <typename T>
struct CandidateDimensionCount<RVals<T>> : RuntimeCandidateDimensionCount<T> {};

template <typename First, typename Other>
consteval auto compatibleCompileVector() -> bool {
  if constexpr (!isCompileVector<Other>)
    return false;
  else
    return compileVectorDimensionCount<Other>() ==
           compileVectorDimensionCount<First>();
}

template <typename First, typename... Remaining>
struct CandidateDimensionCount<CTypes<First, Remaining...>>
    : std::integral_constant<
          std::size_t,
          isCompileVector<First> ? compileVectorDimensionCount<First>() : 1u> {
  static_assert(
      !isCompileVector<First> ||
          (compatibleCompileVector<First, Remaining>() && ...),
      "Every vector in CTypes must have the same number of components.");
};

template <typename Values>
inline constexpr std::size_t candidateDimensionCount =
    CandidateDimensionCount<std::remove_cvref_t<Values>>::value;

template <typename Values, std::size_t Dimension, std::size_t... Candidate>
consteval auto compileComponents(std::index_sequence<Candidate...>) {
  using Tuple = typename Values::values;
  return std::array{
      compileVectorComponent<std::tuple_element_t<Candidate, Tuple>,
                             Dimension>()...};
}

template <typename Values, std::size_t Dimension>
consteval auto compileUniqueComponentCount() -> std::size_t {
  constexpr auto components = compileComponents<Values, Dimension>(
      std::make_index_sequence<Values::size>{});
  auto count = std::size_t{};
  for (std::size_t candidate = 0u; candidate < components.size(); ++candidate) {
    auto duplicate = false;
    for (std::size_t previous = 0u; previous < candidate; ++previous)
      if (components[previous] == components[candidate])
        duplicate = true;
    if (!duplicate)
      ++count;
  }
  return count;
}

template <typename Values, std::size_t Dimension, std::size_t UniqueIndex>
consteval auto compileUniqueComponent() {
  constexpr auto components = compileComponents<Values, Dimension>(
      std::make_index_sequence<Values::size>{});
  auto unique = std::size_t{};
  for (std::size_t candidate = 0u; candidate < components.size(); ++candidate) {
    auto duplicate = false;
    for (std::size_t previous = 0u; previous < candidate; ++previous)
      if (components[previous] == components[candidate])
        duplicate = true;
    if (!duplicate) {
      if (unique == UniqueIndex)
        return components[candidate];
      ++unique;
    }
  }
  throw "A compile-time vector component index is out of range.";
}

template <typename Prototype, auto... Components> struct RebindCompileVector {
  static_assert(alpaka::isCVector_v<Prototype>);
  using type =
      alpaka::CVec<typename Prototype::type,
                   static_cast<typename Prototype::type>(Components)...>;
};

template <typename T, T... Original, auto... Components>
struct RebindCompileVector<std::integer_sequence<T, Original...>,
                           Components...> {
  using type = std::integer_sequence<T, static_cast<T>(Components)...>;
};

template <typename Prototype, auto... Components>
using rebindCompileVector =
    typename RebindCompileVector<Prototype, Components...>::type;

template <typename Values, typename Prototype, typename SelectedIndices>
struct ReconstructCompileVector;

template <typename Values, typename Prototype, std::size_t... Selected>
struct ReconstructCompileVector<Values, Prototype,
                                std::index_sequence<Selected...>> {
  static_assert(sizeof...(Selected) == candidateDimensionCount<Values>);

  template <std::size_t... Dimension>
  static auto make(std::index_sequence<Dimension...>) -> rebindCompileVector<
      Prototype, compileUniqueComponent<Values, Dimension, Selected>()...>;

  using type = decltype(make(
      std::make_index_sequence<candidateDimensionCount<Values>>{}));
};

template <typename Values, typename Prototype, typename SelectedIndices>
using reconstructCompileVector =
    typename ReconstructCompileVector<Values, Prototype, SelectedIndices>::type;

template <typename Sequence, std::size_t Value> struct AppendIndexSequence;

template <std::size_t... Values, std::size_t Value>
struct AppendIndexSequence<std::index_sequence<Values...>, Value> {
  using type = std::index_sequence<Values..., Value>;
};

template <typename Sequence, std::size_t Value>
using appendIndexSequence = typename AppendIndexSequence<Sequence, Value>::type;

} // namespace detail

template <auto Maximum>
using CValsTo = typename detail::MakeCValsRange<0, Maximum, 1>::type;

template <auto Minimum, auto Maximum, auto Step = 1>
using CValsRange =
    typename detail::MakeCValsRange<Minimum, Maximum, Step>::type;

template <FixedString Name, typename Values> struct Tunable {
  static constexpr auto name = Name;
  using values_type = Values;
  Values values;

  constexpr Tunable(TunableName<Name>, Values candidateValues)
      : values(std::move(candidateValues)) {
    static_assert(detail::isRVals<Values> ||
                      detail::isCompileTimeCandidates<Values>,
                  "A tunable needs RVals, CVals, or CTypes.");
  }

  [[nodiscard]] static constexpr auto nameView() -> std::string_view {
    return Name.view();
  }
};

template <FixedString Name, typename Values>
Tunable(TunableName<Name>, Values) -> Tunable<Name, Values>;

template <FixedString Name, typename Values>
using NamedTunable = Tunable<Name, Values>;

template <FixedString Name>
template <typename Values>
[[nodiscard]] constexpr auto TunableName<Name>::operator()(Values values) const
    -> Tunable<Name, std::remove_cvref_t<Values>> {
  using CandidateValues = std::remove_cvref_t<Values>;
  static_assert(detail::isRVals<CandidateValues> ||
                    detail::isCompileTimeCandidates<CandidateValues>,
                "A tunable needs RVals, CVals, or CTypes.");
  return {*this, std::move(values)};
}

struct Named {
  template <typename Name, typename Values>
  [[nodiscard]] constexpr auto operator()(Name name, Values values) const {
    return name(std::move(values));
  }
};

inline constexpr Named named{};

namespace detail {

template <typename Component, typename = void> struct ComponentEntries {
  using type = std::tuple<std::remove_cvref_t<Component>>;
};

template <typename Component>
struct ComponentEntries<
    Component,
    std::void_t<typename std::remove_cvref_t<Component>::entries_type>> {
  using type = typename std::remove_cvref_t<Component>::entries_type;
};

template <typename Component>
using component_entries_t = typename ComponentEntries<Component>::type;

template <typename Component, typename = void> struct ComponentRestrictions {
  using type = std::tuple<>;
};

template <typename Component>
struct ComponentRestrictions<
    Component,
    std::void_t<typename std::remove_cvref_t<Component>::restrictions_type>> {
  using type = typename std::remove_cvref_t<Component>::restrictions_type;
};

template <typename Component>
using component_restrictions_t =
    typename ComponentRestrictions<Component>::type;

template <typename... Tuples>
using tuple_cat_t = decltype(std::tuple_cat(std::declval<Tuples>()...));

template <typename Component>
[[nodiscard]] constexpr auto componentEntries(Component &&component) {
  if constexpr (requires {
                  typename std::remove_cvref_t<Component>::entries_type;
                }) {
    return std::apply(
        [](auto &&...entries) {
          return std::tuple<std::remove_cvref_t<decltype(entries)>...>{
              std::forward<decltype(entries)>(entries)...};
        },
        std::forward<Component>(component).entries());
  } else {
    using Entry = std::remove_cvref_t<Component>;
    static_assert(
        requires { Entry::name; },
        "TunableBundle components must be named tunables or tuning "
        "space fragments.");
    return std::tuple<Entry>{std::forward<Component>(component)};
  }
}

template <typename Component>
[[nodiscard]] constexpr auto componentRestrictions(Component &&component) {
  if constexpr (requires {
                  typename std::remove_cvref_t<Component>::restrictions_type;
                }) {
    return std::apply(
        [](auto &&...restrictions) {
          return std::tuple<std::remove_cvref_t<decltype(restrictions)>...>{
              std::forward<decltype(restrictions)>(restrictions)...};
        },
        std::forward<Component>(component).restrictions());
  } else {
    return std::tuple<>{};
  }
}

template <typename Tuple> struct TupleEntriesAreNamed;

template <typename Entry, typename = void>
struct EntryIsNamed : std::false_type {};

template <typename Entry>
struct EntryIsNamed<Entry, std::void_t<decltype(Entry::name)>>
    : std::true_type {};

template <typename... Entries>
struct TupleEntriesAreNamed<std::tuple<Entries...>>
    : std::bool_constant<(EntryIsNamed<Entries>::value && ...)> {};

} // namespace detail

/**
 * Heterogeneous, statically typed bundle of tunable parameters.
 *
 * A component can be one named tunable or a tuning-space fragment.  Fragment
 * entries and restrictions are flattened into the resulting bundle.
 */
template <typename... Components> class TunableBundle {
public:
  using entries_type =
      detail::tuple_cat_t<detail::component_entries_t<Components>...>;
  using restrictions_type =
      detail::tuple_cat_t<detail::component_restrictions_t<Components>...>;

  explicit constexpr TunableBundle(Components... components)
      : m_entries(
            std::tuple_cat(detail::componentEntries(std::move(components))...)),
        m_restrictions(std::tuple_cat(
            detail::componentRestrictions(std::move(components))...)) {
    static_assert(detail::TupleEntriesAreNamed<entries_type>::value,
                  "TunableBundle entries must be bound to tunable names.");
  }

  [[nodiscard]] constexpr auto entries() const & noexcept
      -> entries_type const & {
    return m_entries;
  }
  [[nodiscard]] constexpr auto entries() && noexcept -> entries_type && {
    return std::move(m_entries);
  }

  [[nodiscard]] constexpr auto restrictions() const & noexcept
      -> restrictions_type const & {
    return m_restrictions;
  }
  [[nodiscard]] constexpr auto restrictions() && noexcept
      -> restrictions_type && {
    return std::move(m_restrictions);
  }

  static constexpr auto size = std::tuple_size_v<entries_type>;

private:
  entries_type m_entries;
  restrictions_type m_restrictions;
};

template <typename... Components>
TunableBundle(Components...) -> TunableBundle<Components...>;

template <typename... Components> using Tunables = TunableBundle<Components...>;

namespace detail {
template <typename T> struct TunablesTraits;

template <typename T> struct IsTunableBundle : std::false_type {};

template <typename... Components>
struct IsTunableBundle<TunableBundle<Components...>> : std::true_type {};

template <typename T>
inline constexpr bool isTunableBundle =
    IsTunableBundle<std::remove_cvref_t<T>>::value;
} // namespace detail

/**
 * A lazy relationship between the values of two named tuning parameters.
 *
 * The ordinary TunableBundle product remains the default.  This wrapper is
 * useful when two otherwise tunable values must move together, for example a
 * frame extent and the matching number of frames for a fixed problem size.
 */
template <FixedString First, FixedString Second, typename Predicate>
class Restriction {
public:
  static constexpr auto first = First;
  static constexpr auto second = Second;

  explicit Restriction(Predicate predicate)
      : m_predicate(std::move(predicate)) {}

  template <alpaka::concepts::VectorOrScalar T_First,
            alpaka::concepts::VectorOrScalar T_Second>
  [[nodiscard]] auto accepts(T_First const &firstValue,
                             T_Second const &secondValue) const -> bool {
    return static_cast<bool>(m_predicate(firstValue, secondValue));
  }

private:
  Predicate m_predicate;
};

template <typename FirstName, typename SecondName, typename Predicate>
  requires requires {
    std::remove_cvref_t<FirstName>::value;
    std::remove_cvref_t<SecondName>::value;
  }
[[nodiscard]] auto restrict(FirstName, SecondName, Predicate predicate) {
  constexpr auto first = std::remove_cvref_t<FirstName>::value;
  constexpr auto second = std::remove_cvref_t<SecondName>::value;
  using Result = Restriction<first, second, std::remove_cvref_t<Predicate>>;
  return Result{std::move(predicate)};
}

template <FixedString Parameter, typename Predicate> class UnaryRestriction {
public:
  static constexpr auto first = Parameter;

  explicit UnaryRestriction(Predicate predicate)
      : m_predicate(std::move(predicate)) {}

  template <alpaka::concepts::VectorOrScalar T>
  [[nodiscard]] auto accepts(T const &value) const -> bool {
    return static_cast<bool>(m_predicate(value));
  }

private:
  Predicate m_predicate;
};

template <typename Name, typename Predicate>
  requires requires { std::remove_cvref_t<Name>::value; }
[[nodiscard]] auto restrict(Name, Predicate predicate) {
  constexpr auto name = std::remove_cvref_t<Name>::value;
  using Result = UnaryRestriction<name, std::remove_cvref_t<Predicate>>;
  return Result{std::move(predicate)};
}

template <typename TunablesType, typename... Restrictions>
class ConstrainedTunables {
public:
  using entries_type = typename TunablesType::entries_type;
  using restrictions_type = std::tuple<Restrictions...>;
  static constexpr auto size = TunablesType::size;

  constexpr ConstrainedTunables(TunablesType tunables,
                                restrictions_type restrictions)
      : m_tunables(std::move(tunables)),
        m_restrictions(std::move(restrictions)) {}

  [[nodiscard]] constexpr auto entries() const & noexcept -> decltype(auto) {
    return m_tunables.entries();
  }
  [[nodiscard]] constexpr auto entries() && noexcept -> decltype(auto) {
    return std::move(m_tunables).entries();
  }

  [[nodiscard]] constexpr auto restrictions() const & noexcept
      -> restrictions_type const & {
    return m_restrictions;
  }
  [[nodiscard]] constexpr auto restrictions() && noexcept
      -> restrictions_type && {
    return std::move(m_restrictions);
  }

private:
  TunablesType m_tunables;
  std::tuple<Restrictions...> m_restrictions;
};

namespace detail {

template <typename TunablesType, typename RestrictionsTuple>
struct ConstrainedTunablesFromTuple;

template <typename TunablesType, typename... Restrictions>
struct ConstrainedTunablesFromTuple<TunablesType, std::tuple<Restrictions...>> {
  using type = ConstrainedTunables<TunablesType, Restrictions...>;
};

template <typename TunablesType, typename RestrictionsTuple>
using constrained_tunables_from_tuple_t =
    typename ConstrainedTunablesFromTuple<TunablesType,
                                          RestrictionsTuple>::type;

} // namespace detail

/** Limit a tuning space with lazy relationships between named parameters. */
template <typename TunablesType, typename... Restrictions>
[[nodiscard]] constexpr auto constrain(TunablesType tunables,
                                       Restrictions... restrictions) {
  static_assert(
      (requires { std::remove_cvref_t<Restrictions>::first; } && ...),
      "constrain accepts relations created with alpakaTune::restrict.");
  using Traits = detail::TunablesTraits<TunablesType>;
  static_assert(
      ((Traits::template has<std::remove_cvref_t<Restrictions>::first> &&
        [] {
          if constexpr (requires { std::remove_cvref_t<Restrictions>::second; })
            return Traits::template has<
                std::remove_cvref_t<Restrictions>::second>;
          else
            return true;
        }()) &&
       ...),
      "Both parameters of every restriction must name tunables in this tuning "
      "space.");
  auto allRestrictions =
      std::tuple_cat(detail::componentRestrictions(std::move(tunables)),
                     std::tuple<std::remove_cvref_t<Restrictions>...>{
                         std::move(restrictions)...});
  using Result =
      detail::constrained_tunables_from_tuple_t<TunablesType,
                                                decltype(allRestrictions)>;
  return TunableBundle{Result{std::move(tunables), std::move(allRestrictions)}};
}

/** Placeholder argument in a prototype alpaka::KernelBundle. */
template <FixedString Name> struct MarkedTunable {
  static constexpr auto name = Name;
  static constexpr bool tunableMarker = true;
};

template <typename Name>
  requires requires { std::remove_cvref_t<Name>::value; }
[[nodiscard]] constexpr auto markTunable(Name) {
  constexpr auto name = std::remove_cvref_t<Name>::value;
  return MarkedTunable<name>{};
}

namespace detail {

template <typename T>
inline constexpr bool isMarkedTunable =
    requires { std::remove_cvref_t<T>::tunableMarker; };

template <FixedString Left, FixedString Right>
inline constexpr bool sameName = Left.view() == Right.view();

template <FixedString Name, typename... Entries>
inline constexpr bool hasEntry = (sameName<Name, Entries::name> || ...);

template <FixedString Name, typename... Entries>
consteval auto findEntryIndexValue() -> std::size_t {
  static_assert(hasEntry<Name, Entries...>,
                "A KernelBundle marker refers to an unknown tunable.");
  constexpr auto matches = std::array{sameName<Name, Entries::name>...};
  for (auto index = std::size_t{}; index < matches.size(); ++index)
    if (matches[index])
      return index;
  return matches.size();
}

template <FixedString Name, typename... Entries>
inline constexpr std::size_t findEntryIndex =
    findEntryIndexValue<Name, Entries...>();

template <FixedString Name, typename... Entries>
using findEntry = std::tuple_element_t<findEntryIndex<Name, Entries...>,
                                       std::tuple<Entries...>>;

template <FixedString Name, typename... Entries>
consteval auto findEntryDimensionOffsetValue() -> std::size_t {
  static_assert(hasEntry<Name, Entries...>,
                "An unknown tunable has no dimension offset.");
  constexpr auto matches = std::array{sameName<Name, Entries::name>...};
  constexpr auto dimensionCounts =
      std::array{candidateDimensionCount<typename Entries::values_type>...};
  auto offset = std::size_t{};
  for (auto index = std::size_t{}; index < matches.size(); ++index) {
    if (matches[index])
      return offset;
    offset += dimensionCounts[index];
  }
  return offset;
}

template <FixedString Name, typename... Entries>
inline constexpr std::size_t findEntryDimensionOffset =
    findEntryDimensionOffsetValue<Name, Entries...>();

template <FixedString Name, typename... Entries>
constexpr auto tupleEntry(std::tuple<Entries...> const &entries)
    -> findEntry<Name, Entries...> const & {
  return std::get<findEntryIndex<Name, Entries...>>(entries);
}

template <typename Tuple> struct TunablesTraitsFromTuple;

template <typename... Entries>
struct TunablesTraitsFromTuple<std::tuple<Entries...>> {
  using entries_type = std::tuple<Entries...>;
  static constexpr std::size_t size = sizeof...(Entries);
  static constexpr std::size_t dimensionCount =
      (candidateDimensionCount<typename Entries::values_type> + ... + 0u);

  template <FixedString Name>
  static constexpr bool has = hasEntry<Name, Entries...>;

  template <FixedString Name> using entry = findEntry<Name, Entries...>;

  template <FixedString Name>
  static constexpr std::size_t index = findEntryIndex<Name, Entries...>;

  template <FixedString Name>
  static constexpr std::size_t dimensionOffset =
      findEntryDimensionOffset<Name, Entries...>;
};

template <typename... Components>
struct TunablesTraits<TunableBundle<Components...>>
    : TunablesTraitsFromTuple<
          typename TunableBundle<Components...>::entries_type> {};

template <typename TunablesType, typename... Restrictions>
struct TunablesTraits<ConstrainedTunables<TunablesType, Restrictions...>>
    : TunablesTraits<TunablesType> {};

} // namespace detail

} // namespace alpakaTune
