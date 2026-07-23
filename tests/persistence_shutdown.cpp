// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#include <alpakaTune/core/Persistence.hpp>

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>

namespace {

constexpr auto schemaVersion = 11;
constexpr auto fingerprint = "shutdown-test";

[[nodiscard]] auto cache(std::string completionReason, int executionCount)
    -> nlohmann::json {
  return {{"fingerprint", fingerprint},
          {"completion_reason", std::move(completionReason)},
          {"execution_count", executionCount}};
}

void writeCompletedHistory(std::filesystem::path const &path) {
  auto history = nlohmann::json{
      {"schema_version", schemaVersion},
      {"contexts", {{fingerprint, cache("all_configurations", 42)}}}};
  std::ofstream output{path};
  output << history;
}

[[nodiscard]] auto stageAndExit(std::filesystem::path const &partialPath,
                                std::filesystem::path const &completedPath)
    -> int {
  std::filesystem::remove(partialPath);
  std::filesystem::remove(completedPath);
  writeCompletedHistory(completedPath);

  auto partialCache = std::make_shared<nlohmann::json>(cache("none", 3));
  alpakaTune::detail::persistenceStore(partialPath)
      ->stageCache(schemaVersion, fingerprint, partialCache);
  (*partialCache)["execution_count"] = 7;
  alpakaTune::detail::persistenceStore(completedPath)
      ->stageCache(schemaVersion, fingerprint,
                   std::make_shared<nlohmann::json>(cache("none", 4)));
  std::exit(EXIT_SUCCESS);
}

[[nodiscard]] auto readHistory(std::filesystem::path const &path)
    -> nlohmann::json {
  auto history = nlohmann::json{};
  auto input = std::ifstream{path};
  input >> history;
  return history;
}

[[nodiscard]] auto verify(std::filesystem::path const &partialPath,
                          std::filesystem::path const &completedPath) -> int {
  auto const partial = readHistory(partialPath);
  auto const &partialCache = partial.at("contexts").at(fingerprint);
  if (partial.at("schema_version") != schemaVersion ||
      partialCache.at("completion_reason") != "none" ||
      partialCache.at("execution_count") != 7)
    return EXIT_FAILURE;

  auto const completed = readHistory(completedPath);
  auto const &completedCache = completed.at("contexts").at(fingerprint);
  if (completedCache.at("completion_reason") != "none" ||
      completedCache.at("execution_count") != 4)
    return EXIT_FAILURE;

  std::filesystem::remove(partialPath);
  std::filesystem::remove(completedPath);
  return EXIT_SUCCESS;
}

} // namespace

auto main(int argc, char **argv) -> int {
  if (argc != 4)
    return EXIT_FAILURE;
  auto const partialPath = std::filesystem::path{argv[2]};
  auto const completedPath = std::filesystem::path{argv[3]};
  if (std::string{argv[1]} == "stage")
    return stageAndExit(partialPath, completedPath);
  if (std::string{argv[1]} == "verify")
    return verify(partialPath, completedPath);
  return EXIT_FAILURE;
}
