// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if ALPAKA_TUNE_HAS_JSON
#include <nlohmann/json.hpp>
#endif

namespace alpakaTune::detail {

/** @brief Process-shared, optionally shutdown-flushed persistence state.
 *
 * Tuners stage complete JSON snapshots in memory. The final shared-store
 * destructor optionally merges only the newest snapshot per fingerprint and
 * performs one atomic file replacement. No launch opens or appends to the
 * history file.
 */
struct CompleteHistoryStore {
  /** @brief Bind the store to optional file-read and shutdown-write policy. */
  CompleteHistoryStore(std::optional<std::filesystem::path> value,
                       bool readValue, bool writeValue)
      : file(std::move(value)), readFile(readValue), writeFile(writeValue) {}

  /** @brief Flush pending snapshots during normal process shutdown.
   *
   * Destructors cannot surface I/O failures, so abnormal or failed shutdown
   * does not guarantee persistence.
   */
  ~CompleteHistoryStore() noexcept {
#if ALPAKA_TUNE_HAS_JSON
    try {
      flush();
    } catch (...) {
      // Static destruction cannot report persistence failures to the caller.
    }
#endif
  }

  /** @brief Write the newest staged snapshots exactly once.
   *
   * Applications should call this after all tuning work has joined so I/O
   * failures can be reported normally. The destructor remains a fallback for
   * callers that do not provide an explicit end-of-run boundary.
   */
  void flush() {
#if ALPAKA_TUNE_HAS_JSON
    std::lock_guard lock{mutex};
    if (file && writeFile && !pendingCaches.empty()) {
      auto caches = std::unordered_map<std::string, nlohmann::json>{};
      caches.reserve(pendingCaches.size());
      for (auto const &[fingerprint, cache] : pendingCaches)
        caches.emplace(fingerprint, *cache);
      mergeAndWrite(schemaVersion, caches);
      pendingCaches.clear();
    }
#endif
  }

#if ALPAKA_TUNE_HAS_JSON
  /** @brief Replace the newest in-memory snapshot for one fingerprint.
   * @throws std::logic_error when one file mixes schema versions.
   */
  void stageCache(int version, std::string const &fingerprint,
                  std::shared_ptr<nlohmann::json> cache) {
    if (!cache)
      throw std::invalid_argument{
          "A staged persistence cache must not be null."};
    std::lock_guard lock{mutex};
    validateSchemaVersion(version);
    pendingCaches[fingerprint] = std::move(cache);
  }

  /** @brief Return the newest staged snapshot without touching the filesystem.
   */
  [[nodiscard]] auto stagedCache(int version, std::string const &fingerprint)
      -> std::shared_ptr<nlohmann::json const> {
    std::lock_guard lock{mutex};
    validateSchemaVersion(version);
    auto const found = pendingCaches.find(fingerprint);
    return found == pendingCaches.end() ? nullptr : found->second;
  }

  /** @brief Copy all staged snapshots for diagnostics and tests. */
  [[nodiscard]] auto stagedCaches(int version)
      -> std::unordered_map<std::string, nlohmann::json> {
    std::lock_guard lock{mutex};
    validateSchemaVersion(version);
    auto caches = std::unordered_map<std::string, nlohmann::json>{};
    caches.reserve(pendingCaches.size());
    for (auto const &[fingerprint, cache] : pendingCaches)
      caches.emplace(fingerprint, *cache);
    return caches;
  }
#endif

  /** Optional absolute normalized path shared by all matching tuners. */
  std::optional<std::filesystem::path> file;
  /** Whether a cache miss may load an existing history file. */
  bool readFile;
  /** Whether staged snapshots are written once during normal shutdown. */
  bool writeFile;
  /** Protects schema selection and the pending snapshot map. */
  std::mutex mutex;

private:
#if ALPAKA_TUNE_HAS_JSON
  /** @brief Lock this store to one persistence schema version. */
  void validateSchemaVersion(int version) {
    if (schemaVersion != 0 && schemaVersion != version)
      throw std::logic_error{
          "One persistence file cannot contain multiple schema versions."};
    schemaVersion = version;
  }

  /** @brief Merge staged contexts and atomically replace the JSON file. */
  void mergeAndWrite(
      int version,
      std::unordered_map<std::string, nlohmann::json> const &caches) const {
    auto const &path = *file;
    auto const parent = path.parent_path();
    if (!parent.empty())
      std::filesystem::create_directories(parent);
    auto store = nlohmann::json::object();
    if (readFile && std::filesystem::exists(path)) {
      std::ifstream input{path};
      if (!(input >> store) || store.value("schema_version", 0) != version ||
          !store.contains("contexts") || !store["contexts"].is_object())
        throw std::runtime_error{"The alpakaTune persistent tuning file is "
                                 "invalid or incompatible."};
    } else {
      store["schema_version"] = version;
      store["contexts"] = nlohmann::json::object();
    }
    for (auto const &[fingerprint, cache] : caches) {
      store["contexts"][fingerprint] = cache;
    }
    auto temporary = path;
    temporary += ".tmp";
    std::ofstream output{temporary};
    if (!output)
      throw std::runtime_error{
          "Unable to write the alpakaTune persistent tuning cache."};
    output << store.dump(2) << '\n';
    output.close();
    std::filesystem::rename(temporary, path);
  }

  int schemaVersion{};
  std::unordered_map<std::string, std::shared_ptr<nlohmann::json>>
      pendingCaches;
#endif
};

/** On-disk schema of the compact sampled history. */
inline constexpr int historySchemaVersion = 1;

#if ALPAKA_TUNE_HAS_JSON
/** @brief Convert one complete staged summary into its compact file context. */
[[nodiscard]] inline auto sampledHistoryContext(nlohmann::json const &cache)
    -> nlohmann::json {
  auto records = std::vector<nlohmann::json>{};
  auto seenConfigurations = std::unordered_set<std::string>{};
  if (auto const found = cache.find("records");
      found != cache.end() && found->is_object()) {
    auto byCandidate = std::vector<nlohmann::json>{};
    byCandidate.reserve(found->size());
    for (auto const &[candidate, record] : found->items()) {
      static_cast<void>(candidate);
      byCandidate.push_back(record);
    }
    std::ranges::sort(byCandidate, [](auto const &left, auto const &right) {
      return left.at("candidate_index").template get<std::size_t>() <
             right.at("candidate_index").template get<std::size_t>();
    });
    records.reserve(byCandidate.size());
    for (auto &record : byCandidate) {
      auto const key = record.at("configuration").dump();
      if (seenConfigurations.insert(key).second)
        records.push_back(std::move(record));
    }
  }
  std::ranges::sort(records, [](auto const &left, auto const &right) {
    auto const leftRuntime =
        left.at("median_runtime_seconds").template get<double>();
    auto const rightRuntime =
        right.at("median_runtime_seconds").template get<double>();
    if (leftRuntime != rightRuntime)
      return leftRuntime < rightRuntime;
    return left.at("configuration").dump() < right.at("configuration").dump();
  });

  auto selected = std::vector<bool>(records.size(), false);
  auto selectedCount = records.size();
  if (auto const count = cache.find("sample_count");
      count != cache.end() && !count->is_null())
    selectedCount = std::min(count->get<std::size_t>(), records.size());
  if (selectedCount == records.size()) {
    std::fill(selected.begin(), selected.end(), true);
  } else {
    auto const forced = std::min<std::size_t>(3u, selectedCount);
    for (std::size_t rank = 0u; rank < forced; ++rank)
      selected.at(rank) = true;
    auto weights = std::vector<double>(records.size(), 0.0);
    for (std::size_t rank = forced; rank < records.size(); ++rank)
      weights.at(rank) = records.size() > 1u
                             ? static_cast<double>(records.size() - 1u - rank)
                             : 0.0;
    auto random =
        std::mt19937_64{cache.value("sampling_seed", std::uint64_t{0u})};
    for (std::size_t slot = forced; slot < selectedCount; ++slot) {
      auto distribution = std::discrete_distribution<std::size_t>{
          weights.begin(), weights.end()};
      auto const rank = distribution(random);
      selected.at(rank) = true;
      weights.at(rank) = 0.0;
    }
  }

  auto result = nlohmann::json{{"configurations", nlohmann::json::array()}};
  for (std::size_t rank = 0u; rank < records.size(); ++rank) {
    if (!selected.at(rank))
      continue;
    auto record = records.at(rank);
    record.erase("candidate_index");
    result["configurations"].push_back(std::move(record));
  }
  if (auto const adapter = cache.find("adapter");
      adapter != cache.end() && adapter->is_object())
    result["adapter"] = *adapter;
  return result;
}
#endif

/** @brief Process-shared compact-history state sampled only at file write. */
struct HistoryStore {
  HistoryStore(std::optional<std::filesystem::path> value, bool readValue,
               bool writeValue)
      : file(std::move(value)), readFile(readValue), writeFile(writeValue) {}

  ~HistoryStore() noexcept {
#if ALPAKA_TUNE_HAS_JSON
    try {
      flush();
    } catch (...) {
      // Static destruction cannot report persistence failures to the caller.
    }
#endif
  }

  /** @brief Sample and write the newest staged histories exactly once.
   *
   * Applications should call this after all tuning work has joined so
   * serialization and I/O failures remain observable.
   */
  void flush() {
#if ALPAKA_TUNE_HAS_JSON
    std::lock_guard lock{mutex};
    if (file && writeFile && !pendingCaches.empty()) {
      auto contexts = std::unordered_map<std::string, nlohmann::json>{};
      contexts.reserve(pendingCaches.size());
      for (auto const &[fingerprint, cache] : pendingCaches)
        contexts.emplace(fingerprint, sampledHistoryContext(*cache));
      mergeAndWrite(contexts);
      pendingCaches.clear();
    }
#endif
  }

#if ALPAKA_TUNE_HAS_JSON
  /** @brief Stage all summaries; sampling is deliberately deferred. */
  void stageCache(std::string const &fingerprint,
                  std::shared_ptr<nlohmann::json> cache) {
    if (!cache)
      throw std::invalid_argument{"A staged history cache must not be null."};
    std::lock_guard lock{mutex};
    pendingCaches[fingerprint] = std::move(cache);
  }

  /** @brief Return the unsampled same-process cache for one fingerprint. */
  [[nodiscard]] auto stagedCache(std::string const &fingerprint)
      -> std::shared_ptr<nlohmann::json const> {
    std::lock_guard lock{mutex};
    auto const found = pendingCaches.find(fingerprint);
    return found == pendingCaches.end() ? nullptr : found->second;
  }
#endif

  std::optional<std::filesystem::path> file;
  bool readFile;
  bool writeFile;
  std::mutex mutex;

private:
#if ALPAKA_TUNE_HAS_JSON
  void mergeAndWrite(
      std::unordered_map<std::string, nlohmann::json> const &contexts) const {
    auto const &path = *file;
    auto const parent = path.parent_path();
    if (!parent.empty())
      std::filesystem::create_directories(parent);
    auto store = nlohmann::json::object();
    if (readFile && std::filesystem::exists(path)) {
      std::ifstream input{path};
      if (!(input >> store) ||
          store.value("schema_version", 0) != historySchemaVersion ||
          !store.contains("contexts") || !store["contexts"].is_object())
        throw std::runtime_error{
            "The alpakaTune compact history is invalid or incompatible."};
    } else {
      store["schema_version"] = historySchemaVersion;
      store["contexts"] = nlohmann::json::object();
    }
    for (auto const &[fingerprint, context] : contexts)
      store["contexts"][fingerprint] = context;
    auto temporary = path;
    temporary += ".tmp";
    std::ofstream output{temporary};
    if (!output)
      throw std::runtime_error{
          "Unable to write the alpakaTune compact history."};
    output << store.dump(2) << '\n';
    output.close();
    std::filesystem::rename(temporary, path);
  }

  std::unordered_map<std::string, std::shared_ptr<nlohmann::json>>
      pendingCaches;
#endif
};

class CompleteHistoryRegistry {
public:
  /** @brief Return the process-wide shared store for one access policy. */
  [[nodiscard]] auto get(std::optional<std::filesystem::path> file,
                         bool readFile, bool writeFile)
      -> std::shared_ptr<CompleteHistoryStore> {
    if (file && file->empty())
      throw std::invalid_argument{"The persistence file must not be empty."};
    if (file)
      file = std::filesystem::absolute(std::move(*file)).lexically_normal();
    auto const key = file
                         ? file->string() + "\nread=" + (readFile ? "1" : "0") +
                               "\nwrite=" + (writeFile ? "1" : "0")
                         : std::string{"<memory>"};
    std::lock_guard lock{mutex};
    auto &store = stores[key];
    if (!store)
      store = std::make_shared<CompleteHistoryStore>(std::move(file), readFile,
                                                     writeFile);
    return store;
  }

  /** @brief Flush every process-shared complete-history store. */
  void flush() {
    auto snapshot = std::vector<std::shared_ptr<CompleteHistoryStore>>{};
    {
      std::lock_guard lock{mutex};
      snapshot.reserve(stores.size());
      for (auto const &[key, store] : stores) {
        static_cast<void>(key);
        snapshot.push_back(store);
      }
    }
    for (auto const &store : snapshot)
      store->flush();
  }

private:
  std::mutex mutex;
  std::unordered_map<std::string, std::shared_ptr<CompleteHistoryStore>> stores;
};

/** @brief Access the process-wide complete-history registry itself. */
inline auto completeHistoryRegistry() -> CompleteHistoryRegistry & {
  static CompleteHistoryRegistry registry;
  return registry;
}

/** @brief Access the process-wide complete-history registry. */
inline auto
completeHistoryStore(std::optional<std::filesystem::path> file = std::nullopt,
                     bool readFile = true, bool writeFile = true)
    -> std::shared_ptr<CompleteHistoryStore> {
  return completeHistoryRegistry().get(std::move(file), readFile, writeFile);
}

class HistoryRegistry {
public:
  [[nodiscard]] auto get(std::optional<std::filesystem::path> file,
                         bool readFile, bool writeFile)
      -> std::shared_ptr<HistoryStore> {
    if (file && file->empty())
      throw std::invalid_argument{"The history file must not be empty."};
    if (file)
      file = std::filesystem::absolute(std::move(*file)).lexically_normal();
    auto const key = file
                         ? file->string() + "\nread=" + (readFile ? "1" : "0") +
                               "\nwrite=" + (writeFile ? "1" : "0")
                         : std::string{"<memory>"};
    std::lock_guard lock{mutex};
    auto &store = stores[key];
    if (!store)
      store =
          std::make_shared<HistoryStore>(std::move(file), readFile, writeFile);
    return store;
  }

  /** @brief Flush every process-shared compact-history store. */
  void flush() {
    auto snapshot = std::vector<std::shared_ptr<HistoryStore>>{};
    {
      std::lock_guard lock{mutex};
      snapshot.reserve(stores.size());
      for (auto const &[key, store] : stores) {
        static_cast<void>(key);
        snapshot.push_back(store);
      }
    }
    for (auto const &store : snapshot)
      store->flush();
  }

private:
  std::mutex mutex;
  std::unordered_map<std::string, std::shared_ptr<HistoryStore>> stores;
};

/** @brief Access the process-wide compact-history registry itself. */
inline auto historyRegistry() -> HistoryRegistry & {
  static HistoryRegistry registry;
  return registry;
}

/** @brief Access the process-wide compact-history registry. */
inline auto
historyStore(std::optional<std::filesystem::path> file = std::nullopt,
             bool readFile = true, bool writeFile = true)
    -> std::shared_ptr<HistoryStore> {
  return historyRegistry().get(std::move(file), readFile, writeFile);
}

} // namespace alpakaTune::detail

namespace alpakaTune {

/** @brief Persist all staged histories at an application end-of-run boundary.
 *
 * Call this only after all threads performing tuned launches have joined.
 * Successful stores are idempotent: their destructors will not write again.
 * Unlike destructor fallback, failures are reported to the caller.
 */
inline void flushPersistence() {
  detail::completeHistoryRegistry().flush();
  detail::historyRegistry().flush();
}

} // namespace alpakaTune
