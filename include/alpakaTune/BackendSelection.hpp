// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <algorithm>
#include <alpaka/alpaka.hpp>
#include <cctype>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>

namespace alpakaTune {

struct BackendSelection {
  std::string api;
  std::string deviceKind;
  std::string executor;
};

inline auto backendSelection() -> std::optional<BackendSelection> & {
  static auto selection = std::optional<BackendSelection>{};
  return selection;
}

inline auto normalizeBackendName(std::string value) -> std::string {
  value.erase(std::remove_if(value.begin(), value.end(),
                             [](unsigned char character) {
                               return std::isspace(character);
                             }),
              value.end());
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  return value;
}

inline auto consumeBackendOptions(int &argc, char **argv) -> bool {
  auto selection = BackendSelection{};
  auto hasSelection = false;
  auto write = 1;
  for (auto read = 1; read < argc; ++read) {
    auto const argument = std::string_view{argv[read]};
    auto option = std::string_view{};
    auto value = std::string_view{};
    if (argument == "--backend" || argument == "--executor") {
      option = argument;
      if (++read >= argc) {
        std::cerr << argument << " requires an argument.\n";
        return false;
      }
      value = argv[read];
    } else if (argument.starts_with("--backend=") ||
               argument.starts_with("--executor=")) {
      auto const separator = argument.find('=');
      option = argument.substr(0, separator);
      value = argument.substr(separator + 1u);
    } else {
      argv[write++] = argv[read];
      continue;
    }

    if (value.empty()) {
      std::cerr << option << " requires a non-empty argument.\n";
      return false;
    }
    if (option == "--backend") {
      auto const separator = value.find(':');
      if (separator == std::string_view::npos || separator == 0u ||
          separator + 1u >= value.size() ||
          value.find(':', separator + 1u) != std::string_view::npos) {
        std::cerr << "--backend must have the form api:deviceKind (for example "
                     "cuda:nvidiaGpu).\n";
        return false;
      }
      selection.api =
          normalizeBackendName(std::string{value.substr(0u, separator)});
      selection.deviceKind =
          normalizeBackendName(std::string{value.substr(separator + 1u)});
      hasSelection = true;
    } else {
      selection.executor = normalizeBackendName(std::string{value});
      hasSelection = true;
    }
  }
  argv[write] = nullptr;
  argc = write;
  backendSelection() =
      hasSelection ? std::optional<BackendSelection>{std::move(selection)}
                   : std::nullopt;
  return true;
}

template <alpaka::concepts::BackendSpec Backend>
inline auto backendSelected(Backend const &backend) -> bool {
  auto const &selection = backendSelection();
  if (!selection)
    return true;
  auto const device = alpaka::onHost::DeviceSpec{backend};
  auto const api = normalizeBackendName(device.getApi().getName());
  auto const deviceKind =
      normalizeBackendName(device.getDeviceKind().getName());
  auto const executor = normalizeBackendName(
      alpaka::onHost::demangledName(alpaka::getExecutor(backend)));
  auto const matches = [](std::string const &actual,
                          std::string const &requested) {
    return actual == requested || actual.ends_with("::" + requested) ||
           actual.ends_with(requested);
  };
  return (selection->api.empty() || matches(api, selection->api)) &&
         (selection->deviceKind.empty() ||
          matches(deviceKind, selection->deviceKind)) &&
         (selection->executor.empty() ||
          matches(executor, selection->executor));
}

} // namespace alpakaTune
