// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

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
struct PersistenceStore {
  /** @brief Bind the store to optional file-read and shutdown-write policy. */
  PersistenceStore(std::optional<std::filesystem::path> value,
                   bool readValue, bool writeValue)
      : file(std::move(value)), readFile(readValue), writeFile(writeValue) {}

  /** @brief Flush pending snapshots during normal process shutdown.
   *
   * Destructors cannot surface I/O failures, so abnormal or failed shutdown
   * does not guarantee persistence.
   */
  ~PersistenceStore() noexcept {
#if ALPAKA_TUNE_HAS_JSON
    try {
      std::lock_guard lock{mutex};
      if (file && writeFile && !pendingCaches.empty()) {
        auto caches = std::unordered_map<std::string, nlohmann::json>{};
        caches.reserve(pendingCaches.size());
        for (auto const &[fingerprint, cache] : pendingCaches)
          caches.emplace(fingerprint, *cache);
        mergeAndWrite(schemaVersion, caches);
        pendingCaches.clear();
      }
    } catch (...) {
      // Static destruction cannot report persistence failures to the caller.
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

class PersistenceRegistry {
public:
  /** @brief Return the process-wide shared store for one access policy. */
  [[nodiscard]] auto get(std::optional<std::filesystem::path> file,
                         bool readFile, bool writeFile)
      -> std::shared_ptr<PersistenceStore> {
    if (file && file->empty())
      throw std::invalid_argument{"The persistence file must not be empty."};
    if (file)
      file = std::filesystem::absolute(std::move(*file)).lexically_normal();
    auto const key =
        file ? file->string() + "\nread=" + (readFile ? "1" : "0") +
                   "\nwrite=" + (writeFile ? "1" : "0")
             : std::string{"<memory>"};
    std::lock_guard lock{mutex};
    auto &store = stores[key];
    if (!store)
      store = std::make_shared<PersistenceStore>(
          std::move(file), readFile, writeFile);
    return store;
  }

private:
  std::mutex mutex;
  std::unordered_map<std::string, std::shared_ptr<PersistenceStore>> stores;
};

/** @brief Access the process-wide persistence registry. */
inline auto persistenceStore(
    std::optional<std::filesystem::path> file = std::nullopt,
    bool readFile = true, bool writeFile = true)
    -> std::shared_ptr<PersistenceStore> {
  static PersistenceRegistry registry;
  return registry.get(std::move(file), readFile, writeFile);
}

} // namespace alpakaTune::detail
