/* Copyright 2025 Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once
#include <random>

namespace tune::strategy::helper {

class StdMTRandomDevice {
public:
  static std::mt19937 &get() {
    static StdMTRandomDevice instance;
    return instance.rng_;
  }

private:
  StdMTRandomDevice() {
    std::random_device rd;
    rng_ = std::mt19937(rd());
  }

  std::mt19937 rng_;
};
} // namespace tune::strategy::helper
