// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#if ALPAKA_TUNE_HAS_JSON
#include <nlohmann/json.hpp>
#endif

namespace alpakaTune::detail {

struct PersistenceStore {
  explicit PersistenceStore(std::filesystem::path value)
      : file(std::move(value)) {}

  ~PersistenceStore() noexcept {
#if ALPAKA_TUNE_HAS_JSON
    try {
      std::lock_guard lock{mutex};
      if (!pendingCaches.empty()) {
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
  void stageCache(int version, std::string const &fingerprint,
                  std::shared_ptr<nlohmann::json> cache) {
    if (!cache)
      throw std::invalid_argument{
          "A staged persistence cache must not be null."};
    std::lock_guard lock{mutex};
    validateSchemaVersion(version);
    if (completedFingerprints.contains(fingerprint))
      return;
    pendingCaches[fingerprint] = std::move(cache);
  }

  void commitCache(int version, std::string const &fingerprint,
                   nlohmann::json cache) {
    std::lock_guard lock{mutex};
    validateSchemaVersion(version);
    auto caches = std::unordered_map<std::string, nlohmann::json>{};
    caches.emplace(fingerprint, std::move(cache));
    mergeAndWrite(version, caches);
    pendingCaches.erase(fingerprint);
    completedFingerprints.insert(fingerprint);
  }
#endif

  std::filesystem::path file;
  std::mutex mutex;

private:
#if ALPAKA_TUNE_HAS_JSON
  void validateSchemaVersion(int version) {
    if (schemaVersion != 0 && schemaVersion != version)
      throw std::logic_error{
          "One persistence file cannot contain multiple schema versions."};
    schemaVersion = version;
  }

  [[nodiscard]] static auto isCompleted(nlohmann::json const &cache) -> bool {
    if (!cache.is_object())
      return false;
    auto const reason = cache.value("completion_reason", std::string{"none"});
    return reason == "all_configurations" || reason == "maximum_executions" ||
           reason == "maximum_retired_configurations";
  }

  void mergeAndWrite(
      int version,
      std::unordered_map<std::string, nlohmann::json> const &caches) const {
    auto const parent = file.parent_path();
    if (!parent.empty())
      std::filesystem::create_directories(parent);
    auto store = nlohmann::json::object();
    if (std::filesystem::exists(file)) {
      std::ifstream input{file};
      if (!(input >> store) || store.value("schema_version", 0) != version ||
          !store.contains("contexts") || !store["contexts"].is_object())
        throw std::runtime_error{"The alpakaTune persistent tuning file is "
                                 "invalid or incompatible."};
    } else {
      store["schema_version"] = version;
      store["contexts"] = nlohmann::json::object();
    }
    for (auto const &[fingerprint, cache] : caches) {
      auto const existing = store["contexts"].find(fingerprint);
      if (existing != store["contexts"].end() && isCompleted(*existing))
        continue;
      store["contexts"][fingerprint] = cache;
    }
    auto temporary = file;
    temporary += ".tmp";
    std::ofstream output{temporary};
    if (!output)
      throw std::runtime_error{
          "Unable to write the alpakaTune persistent tuning cache."};
    output << store.dump(2) << '\n';
    output.close();
    std::filesystem::rename(temporary, file);
  }

  int schemaVersion{};
  std::unordered_map<std::string, std::shared_ptr<nlohmann::json>>
      pendingCaches;
  std::unordered_set<std::string> completedFingerprints;
#endif
};

class PersistenceRegistry {
public:
  [[nodiscard]] auto get(std::filesystem::path file)
      -> std::shared_ptr<PersistenceStore> {
    if (file.empty())
      throw std::invalid_argument{"The persistence file must not be empty."};
    auto absolute =
        std::filesystem::absolute(std::move(file)).lexically_normal();
    auto const key = absolute.string();
    std::lock_guard lock{mutex};
    auto &store = stores[key];
    if (!store)
      store = std::make_shared<PersistenceStore>(std::move(absolute));
    return store;
  }

private:
  std::mutex mutex;
  std::unordered_map<std::string, std::shared_ptr<PersistenceStore>> stores;
};

inline auto persistenceStore(std::filesystem::path file)
    -> std::shared_ptr<PersistenceStore> {
  static PersistenceRegistry registry;
  return registry.get(std::move(file));
}

} // namespace alpakaTune::detail
