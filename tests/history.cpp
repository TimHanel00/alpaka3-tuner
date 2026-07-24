// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#include <alpakaTune/core/Persistence.hpp>
#include <alpakaTune/store/RuntimeHistory.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

namespace {

[[nodiscard]] auto stagedHistory(std::size_t sampleCount) -> nlohmann::json {
  auto cache = nlohmann::json{{"records", nlohmann::json::object()},
                              {"sampling_seed", 42u},
                              {"sample_count", sampleCount},
                              {"adapter",
                               {{"model_digest", "digest"},
                                {"state_version", 1},
                                {"coefficients", {1.0, 2.0}},
                                {"observations_since_update", 1},
                                {"update_count", 2},
                                {"observations", nlohmann::json::array()}}}};
  for (std::size_t candidate = 0u; candidate < 6u; ++candidate)
    cache["records"][std::to_string(candidate)] = {
        {"candidate_index", candidate},
        {"configuration", {{"value", candidate}}},
        {"median_runtime_seconds", static_cast<double>(candidate + 1u)},
        {"measurement_count", candidate + 2u}};
  return cache;
}

} // namespace

auto main() -> int {
#if ALPAKA_TUNE_HAS_JSON
  auto const four =
      alpakaTune::detail::sampledHistoryContext(stagedHistory(4u));
  if (four != alpakaTune::detail::sampledHistoryContext(stagedHistory(4u)) ||
      four.at("configurations").size() != 4u ||
      !four.contains("adapter"))
    return EXIT_FAILURE;
  auto previousRuntime = 0.0;
  for (std::size_t rank = 0u; rank < four.at("configurations").size(); ++rank) {
    auto const &record = four.at("configurations").at(rank);
    auto const runtime = record.at("median_runtime_seconds").get<double>();
    if (record.contains("candidate_index") || runtime < previousRuntime)
      return EXIT_FAILURE;
    previousRuntime = runtime;
    if (rank < 3u &&
        record.at("configuration").at("value").get<std::size_t>() != rank)
      return EXIT_FAILURE;
  }
  if (four.at("configurations")
          .back()
          .at("configuration")
          .at("value")
          .get<std::size_t>() == 5u)
    return EXIT_FAILURE;

  auto const two =
      alpakaTune::detail::sampledHistoryContext(stagedHistory(2u));
  if (two.at("configurations").size() != 2u ||
      two.at("configurations").at(0).at("configuration").at("value") != 0u ||
      two.at("configurations").at(1).at("configuration").at("value") != 1u)
    return EXIT_FAILURE;

  auto allCache = stagedHistory(6u);
  allCache["sample_count"] = nullptr;
  auto const all = alpakaTune::detail::sampledHistoryContext(allCache);
  if (all.at("configurations").size() != 6u)
    return EXIT_FAILURE;

  auto const path =
      std::filesystem::temp_directory_path() / "alpakaTune-history-test.json";
  std::filesystem::remove(path);
  {
    auto store = alpakaTune::detail::HistoryStore{path, false, true};
    store.stageCache("context",
                     std::make_shared<nlohmann::json>(stagedHistory(4u)));
  }
  auto persisted = nlohmann::json{};
  auto input = std::ifstream{path};
  input >> persisted;
  if (persisted.at("schema_version") !=
          alpakaTune::detail::historySchemaVersion ||
      persisted.at("contexts")
              .at("context")
              .at("configurations")
              .size() != 4u ||
      !persisted.at("contexts").at("context").contains("adapter"))
    return EXIT_FAILURE;

  {
    auto store = alpakaTune::detail::HistoryStore{path, true, false};
    store.stageCache("read-only",
                     std::make_shared<nlohmann::json>(stagedHistory(2u)));
  }
  auto afterReadOnly = nlohmann::json{};
  auto readOnlyInput = std::ifstream{path};
  readOnlyInput >> afterReadOnly;
  if (afterReadOnly != persisted)
    return EXIT_FAILURE;

  {
    auto store = alpakaTune::detail::HistoryStore{path, true, true};
    store.stageCache("second",
                     std::make_shared<nlohmann::json>(stagedHistory(2u)));
  }
  auto merged = nlohmann::json{};
  auto mergedInput = std::ifstream{path};
  mergedInput >> merged;
  if (!merged.at("contexts").contains("context") ||
      !merged.at("contexts").contains("second"))
    return EXIT_FAILURE;
  std::filesystem::remove(path);
#endif

  auto history = alpakaTune::detail::RuntimeHistory{
      {.historyWindowSize = 3u, .automaticRetirement = false}};
  history.restoreSummary(1.25, 10u);
  if (history.samples().size() != 3u ||
      history.currentRunSampleCount() != 0u ||
      history.statistics().sampleCount != 3u ||
      history.statistics().estimate() != 1.25)
    return EXIT_FAILURE;
  return EXIT_SUCCESS;
}
