// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/onHost/Device.hpp>
#include <alpaka/onHost/internal/interface.hpp>

namespace alpakaTune {

/**
 * @brief Obtain Alpaka's current FrameSpec-to-ThreadSpec mapping.
 *
 * This intentionally uses Alpaka3's internal AdjustThreadSpec extension point.
 * It is provided so a tuning application can inspect or persist the exact
 * launch shape selected for a FrameSpec and kernel bundle. The mapping can be
 * customised by specialising Alpaka's corresponding internal operation.
 */
template <typename Device, typename FrameSpec, typename KernelBundle>
[[nodiscard]] auto deriveThreadSpec(Device const &device,
                                    FrameSpec const &frameSpec,
                                    KernelBundle const &kernelBundle) {
  return alpaka::onHost::internal::adjustThreadSpec(*device.get(), frameSpec,
                                                    kernelBundle);
}

} // namespace alpakaTune
