// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0
#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/core/Sycl.hpp>
#include <catch2/catch_test_macros.hpp>

#include <concepts>
#include <functional>
#include <type_traits>
#include <utility>

namespace alpakaTune::test {
template <typename TData>
void runIfSupported(alpaka::onHost::concepts::Device auto device,
                    std::invocable auto &&action) {
#if ALPAKA_LANG_ONEAPI
  if constexpr (std::is_same_v<TData, double> &&
                ALPAKA_TYPEOF(device.getApi()){} == alpaka::api::oneApi) {
    if (device.getNativeHandle()
            .first.template get_info<sycl::info::device::double_fp_config>()
            .empty()) {
      SUCCEED(alpaka::onHost::getName(device)
              << " does not support double precision; skipping this device "
                 "test. Set IGC_EnableDPEmulation=1 and "
                 "OverrideDefaultFP64Settings=1 to enable emulation.");
      return;
    }
  }
#endif
  std::invoke(std::forward<decltype(action)>(action));
}
} // namespace alpakaTune::test
