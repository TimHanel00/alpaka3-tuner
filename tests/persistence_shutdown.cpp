// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#include <alpakaTune/core/Persistence.hpp>

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace {

constexpr auto schemaVersion = 11;
constexpr auto fingerprint = "shutdown-test";
constexpr auto staleFingerprint = "stale-test";
constexpr auto freshFingerprint = "fresh-test";

[[nodiscard]] auto cache(std::string_view cacheFingerprint,
                         std::string completionReason, int executionCount)
    -> nlohmann::json {
  return {{"fingerprint", cacheFingerprint},
          {"completion_reason", std::move(completionReason)},
          {"execution_count", executionCount}};
}

[[nodiscard]] auto completedHistory() -> nlohmann::json {
  return {{"schema_version", schemaVersion},
          {"contexts",
           {{fingerprint, cache(fingerprint, "all_configurations", 42)},
            {staleFingerprint,
             cache(staleFingerprint, "all_configurations", 99)}}}};
}

void writeCompletedHistory(std::filesystem::path const &path) {
  std::ofstream output{path};
  output << completedHistory();
}

[[nodiscard]] auto stageAndExit(std::filesystem::path const &partialPath,
                                std::filesystem::path const &completedPath,
                                std::filesystem::path const &writeOnlyPath,
                                std::filesystem::path const &readOnlyPath)
    -> int {
  std::filesystem::remove(partialPath);
  std::filesystem::remove(completedPath);
  std::filesystem::remove(writeOnlyPath);
  std::filesystem::remove(readOnlyPath);
  writeCompletedHistory(completedPath);
  writeCompletedHistory(writeOnlyPath);
  writeCompletedHistory(readOnlyPath);

  auto partialCache =
      std::make_shared<nlohmann::json>(cache(fingerprint, "none", 3));
  alpakaTune::detail::persistenceStore(partialPath)
      ->stageCache(schemaVersion, fingerprint, partialCache);
  (*partialCache)["execution_count"] = 7;
  alpakaTune::detail::persistenceStore(completedPath)
      ->stageCache(schemaVersion, fingerprint,
                   std::make_shared<nlohmann::json>(
                       cache(fingerprint, "none", 4)));
  alpakaTune::detail::persistenceStore(writeOnlyPath, false, true)
      ->stageCache(schemaVersion, freshFingerprint,
                   std::make_shared<nlohmann::json>(
                       cache(freshFingerprint, "none", 5)));
  alpakaTune::detail::persistenceStore(readOnlyPath, true, false)
      ->stageCache(schemaVersion, freshFingerprint,
                   std::make_shared<nlohmann::json>(
                       cache(freshFingerprint, "none", 6)));
  std::exit(EXIT_SUCCESS);
}

[[nodiscard]] auto readHistory(std::filesystem::path const &path)
    -> nlohmann::json {
  auto history = nlohmann::json{};
  auto input = std::ifstream{path};
  input >> history;
  return history;
}

[[nodiscard]] auto readText(std::filesystem::path const &path) -> std::string {
  auto input = std::ifstream{path};
  auto content = std::ostringstream{};
  content << input.rdbuf();
  return content.str();
}

[[nodiscard]] auto verify(std::filesystem::path const &partialPath,
                          std::filesystem::path const &completedPath,
                          std::filesystem::path const &writeOnlyPath,
                          std::filesystem::path const &readOnlyPath) -> int {
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

  auto const writeOnly = readHistory(writeOnlyPath);
  auto const &writeOnlyContexts = writeOnly.at("contexts");
  if (writeOnlyContexts.size() != 1u ||
      !writeOnlyContexts.contains(freshFingerprint) ||
      writeOnlyContexts.at(freshFingerprint).at("execution_count") != 5)
    return EXIT_FAILURE;

  if (readText(readOnlyPath) != completedHistory().dump())
    return EXIT_FAILURE;

  std::filesystem::remove(partialPath);
  std::filesystem::remove(completedPath);
  std::filesystem::remove(writeOnlyPath);
  std::filesystem::remove(readOnlyPath);
  return EXIT_SUCCESS;
}

} // namespace

auto main(int argc, char **argv) -> int {
  if (argc != 6)
    return EXIT_FAILURE;
  auto const partialPath = std::filesystem::path{argv[2]};
  auto const completedPath = std::filesystem::path{argv[3]};
  auto const writeOnlyPath = std::filesystem::path{argv[4]};
  auto const readOnlyPath = std::filesystem::path{argv[5]};
  if (std::string{argv[1]} == "stage")
    return stageAndExit(partialPath, completedPath, writeOnlyPath, readOnlyPath);
  if (std::string{argv[1]} == "verify")
    return verify(partialPath, completedPath, writeOnlyPath, readOnlyPath);
  return EXIT_FAILURE;
}
