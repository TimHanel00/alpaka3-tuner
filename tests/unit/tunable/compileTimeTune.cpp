
/* Copyright 2025 Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */
#include <alpakaTune/tunable/tunables.hpp>

#include <alpakaTune/utils/compileTimeTemplates.hpp>

#include <catch2/catch_test_macros.hpp>

struct nonTrivial {};

template <typename CVec_> struct TestKernel {
  using Param1 = CVec_;
  TestKernel() = default;

  ALPAKA_FN_HOST void operator()(auto const &) const {};

  static auto getValue() { return CVec_{}; }
};

template <typename T, typename CVec_, typename T2> struct TestKernelThreeDim {
  using Param1 = T;
  using Param2 = CVec_;
  using Param3 = T2;
  TestKernelThreeDim() = default;

  ALPAKA_FN_HOST void operator()(auto const &) const {};

  static auto getValue() { return CVec_{}; }
};

template <typename T, typename CVec_, typename CVec2_> struct TestKernelMD {
  using Param1 = T;
  using Param2 = CVec_;
  using Param3 = CVec2_;
  TestKernelMD() = default;

  ALPAKA_FN_HOST void operator()(auto const &) const {};

  static auto getValue_1() { return CVec_{}; }

  static auto getValue_2() { return CVec2_{}; }
};

struct foo {
  template <typename T_Other> bool operator==(T_Other const &) {
    if constexpr (std::is_same_v<foo, std::remove_cvref_t<T_Other>>) {
      return true;
    } else {
      return false;
    }
  }
};

struct bar {
  template <typename T_Other> bool operator==(T_Other const &) {
    if constexpr (std::is_same_v<bar, std::remove_cvref_t<T_Other>>) {
      return true;
    } else {
      return false;
    }
  }
};

struct baz {
  template <typename T_Other> bool operator==(T_Other const &) {
    if constexpr (std::is_same_v<baz, std::remove_cvref_t<T_Other>>) {
      return true;
    } else {
      return false;
    }
  }
};

template <typename T, typename A, typename B> struct TestKernelArbitrary {
  using Param1 = T;
  using Param2 = A;
  using Param3 = B;

  TestKernelArbitrary() = default;

  ALPAKA_FN_HOST void operator()(auto const &) const {}

  static auto getValue_1() { return T{}; }

  static auto getValue_2() { return A{}; }

  static auto getValue_3() { return B{}; }
};

namespace aTune::trait {
template <typename T> struct CompileTimeTuneableTrait<TestKernel<T>> {
  static constexpr auto tuned_indices =
      alpaka::CVec<std::size_t, static_cast<std::size_t>(0)>{};
  using t = typename T::type;

  static auto tuneAbleDefinitions() {
    auto tune1 = aTune::CTunable<static_cast<std::size_t>(0),
                                 alpaka::CVec<t, 1>, alpaka::CVec<t, 2>,
                                 alpaka::CVec<t, 8>, alpaka::CVec<t, 10>>{};
    return std::tuple{tune1};
  }
};

template <typename T, typename CVec_type, typename T2>
struct CompileTimeTuneableTrait<TestKernelThreeDim<T, CVec_type, T2>> {
  // change Index of template parameter
  static constexpr auto tuned_indices =
      alpaka::CVec<std::size_t, static_cast<std::size_t>(1)>{};
  using t = typename CVec_type::type;

  static auto tuneAbleDefinitions() {
    auto tune1 = aTune::CTunable<static_cast<std::size_t>(6),
                                 alpaka::CVec<t, 1>, alpaka::CVec<t, 2>,
                                 alpaka::CVec<t, 8>, alpaka::CVec<t, 10>>{};
    return std::tuple{tune1};
  }
};

template <typename T, typename CVec_type, typename CVec_type2>
struct CompileTimeTuneableTrait<TestKernelMD<T, CVec_type, CVec_type2>> {
  static constexpr auto tuned_indices =
      alpaka::CVec<std::size_t, static_cast<std::size_t>(1),
                   static_cast<std::size_t>(2)>{};
  using t = typename CVec_type::type;

  static auto tuneAbleDefinitions() {
    auto tune1 = aTune::CTunable<static_cast<std::size_t>(6),
                                 alpaka::CVec<t, 1>, alpaka::CVec<t, 2>,
                                 alpaka::CVec<t, 8>, alpaka::CVec<t, 10>>{};
    auto tune2 =
        aTune::CTunable<static_cast<std::size_t>(12),
                        alpaka::CVec<uint32_t, 15>, alpaka::CVec<uint32_t, 20>,
                        alpaka::CVec<uint32_t, 200>>{};
    return std::tuple{tune1, tune2};
  }
};

template <typename T, typename A, typename B>
struct CompileTimeTuneableTrait<TestKernelArbitrary<T, A, B>> {
  static constexpr auto tuned_indices = alpaka::CVec<std::size_t, 0u, 1u, 2u>{};

  static auto tuneAbleDefinitions() {
    auto tune1 = aTune::CTunable<140u /***ID***/, foo, bar, baz>{};
    auto tune2 = aTune::CTunable<140u /***ID***/, bar, baz, foo>{};
    auto tune3 =
        aTune::CTunable<140u /***ID***/, alpaka::CVec<uint32_t, 1u, 2u>,
                        alpaka::CVec<uint32_t, 3u, 2u>,
                        alpaka::CVec<uint32_t, 4u, 5u>>{};
    return std::tuple{tune1, tune2, tune3};
  }
};
} // namespace aTune::trait

TEST_CASE("parseCompileTimeTuneables", "[KernelVariantGeneration]") {
  TestKernel<alpaka::CVec<uint32_t, 0>> tuned{};
  [[maybe_unused]] auto bundle = alpaka::KernelBundle{tuned};
  using kernelFn = typename decltype(bundle)::KernelFn;
  static_assert(aTune::trait::hasUserDefinedCTuneable<kernelFn>::value);
  [[maybe_unused]] static auto variants =
      aTune::internal::compileTimeHelpers::RegisteredCTuneables<
          std::decay_t<kernelFn>>::T_KernelVariants{};
  auto vals =
      std::tuple{alpaka::CVec<uint32_t, 1>{}, alpaka::CVec<uint32_t, 2>{},
                 alpaka::CVec<uint32_t, 8>{}, alpaka::CVec<uint32_t, 10>{}};
  aTune::meta::forEachEnumerate(variants, [&]<typename T0>(T0 const &val,
                                                           auto idx) {
    static_assert(
        std::is_convertible_v<typename T0::Param1, alpaka::CVec<uint32_t, 18>>);
    aTune::meta::visitIndex(
        idx, vals, [&](auto const &val2) { CHECK(val.getValue() == val2); });
  });
};

TEST_CASE("parseCompileTimeTuneableChangeIndex",
          "[KernelVariantGenerationWithChangedIndex]") {
  TestKernelThreeDim<nonTrivial, alpaka::CVec<uint32_t, 12>, float_t> tuned{};
  [[maybe_unused]] auto bundle = alpaka::KernelBundle{tuned};
  using kernelFn = typename decltype(bundle)::KernelFn;
  static_assert(aTune::trait::hasUserDefinedCTuneable<kernelFn>::value);
  static auto variants =
      aTune::internal::compileTimeHelpers::RegisteredCTuneables<
          std::decay_t<kernelFn>>::T_KernelVariants{};
  auto vals =
      std::tuple{alpaka::CVec<uint32_t, 1>{}, alpaka::CVec<uint32_t, 2>{},
                 alpaka::CVec<uint32_t, 8>{}, alpaka::CVec<uint32_t, 10>{}};
  aTune::meta::forEachEnumerate(variants, [&]<typename T0>(T0 const &val,
                                                           auto idx) {
    static_assert(std::is_same_v<typename T0::Param1, nonTrivial>);
    static_assert(
        std::is_convertible_v<typename T0::Param2, alpaka::CVec<uint32_t, 18>>);
    static_assert(std::is_same_v<typename T0::Param3, float_t>);
    aTune::meta::visitIndex(
        idx, vals, [&](auto const &val2) { CHECK(val.getValue() == val2); });
  });
};

TEST_CASE("parseCompileTimeTuneableMultiDim",
          "[KernelVariantGenerationWithMultipleDimensions]") {
  TestKernelMD<nonTrivial, alpaka::CVec<uint32_t, 12>,
               alpaka::CVec<uint32_t, 11>>
      tuned{};
  [[maybe_unused]] auto bundle = alpaka::KernelBundle{tuned};
  using kernelFn = typename decltype(bundle)::KernelFn;
  static_assert(aTune::trait::hasUserDefinedCTuneable<kernelFn>::value);

  constexpr auto testIndicies =
      std::tuple{std::array<uint32_t, 2>{0, 2}, std::array<uint32_t, 2>{1, 1},
                 std::array<uint32_t, 2>{3, 0}};
  constexpr auto expectedValues_1 =
      std::tuple{alpaka::CVec<uint32_t, 1>{}, alpaka::CVec<uint32_t, 2>{},
                 alpaka::CVec<uint32_t, 10>{}};
  constexpr auto expectedValues_2 =
      std::tuple{alpaka::CVec<uint32_t, 200>{}, alpaka::CVec<uint32_t, 20>{},
                 alpaka::CVec<uint32_t, 15>{}};
  aTune::meta::forEachEnumerate(
      testIndicies, [&]<std::size_t I>(auto const &mDim_Idx) {
        aTune::internal::compileTimeHelpers::runtime_Kernel_dispatch<kernelFn>(
            mDim_Idx, [&]<typename T_KernelBundle>(T_KernelBundle &&element) {
              using T_raw = std::remove_cvref_t<T_KernelBundle>;
              CHECK(std::get<I>(expectedValues_1) == element.getValue_1());
              CHECK(std::get<I>(expectedValues_2) == element.getValue_2());
            });
      });
};

template <typename T> struct Dummy;

TEST_CASE("runtime_Kernel_dispatch for arbitrary 3D kernel",
          "[KernelVariantGeneration]") {
  using kernelFn = TestKernelArbitrary<nonTrivial, foo, bar>;

  static_assert(aTune::trait::hasUserDefinedCTuneable<kernelFn>::value);

  // Now 3-dimensional indices for three parameters
  constexpr auto testIndices = std::tuple{std::array<std::size_t, 3>{0, 1, 2},
                                          std::array<std::size_t, 3>{1, 2, 0},
                                          std::array<std::size_t, 3>{2, 0, 1}};
  // constexpr auto testIndices = std::tuple{std::array<std::size_t, 3>{0, 1,
  // 2}};
  //  auto tune1 = aTune::CTunable<140u /***ID***/, foo, bar, baz>{};
  //  auto tune2 = aTune::CTunable<140u /***ID***/, bar, baz, foo>{};
  //  auto tune3 = aTune::CTunable<
  //      140u /***ID***/,
  //      alpaka::CVec<uint32_t, 1u, 2u>,
  //      alpaka::CVec<uint32_t, 3u, 2u>,
  //      alpaka::CVec<uint32_t, 4u, 5u>>{};
  constexpr auto expectedValues_1 = std::tuple{foo{}, bar{}, baz{}};
  constexpr auto expectedValues_2 = std::tuple{baz{}, foo{}, bar{}};
  constexpr auto expectedValues_3 = std::tuple{
      alpaka::CVec<uint32_t, 4u, 5u>{}, alpaka::CVec<uint32_t, 1u, 2u>{},
      alpaka::CVec<uint32_t, 3u, 2u>{}};

  aTune::meta::forEachEnumerate(
      testIndices, [&]<std::size_t I>(auto const &mdimIdx) {
        aTune::internal::compileTimeHelpers::runtime_Kernel_dispatch<kernelFn>(
            mdimIdx, [&]<typename T_KernelBundle>(T_KernelBundle &&element) {
              CHECK(std::get<I>(expectedValues_1) == element.getValue_1());

              CHECK(std::get<I>(expectedValues_2) == element.getValue_2());

              CHECK(std::get<I>(expectedValues_3) == element.getValue_3());
            });
      });
}
