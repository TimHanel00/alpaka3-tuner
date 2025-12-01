
/* Copyright 2025 Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */
#include <alpaka/meta/meta.hpp>
#include <tune/config/Config.hpp>
#include <tune/core/peripherals/ConfigQueue.hpp>
#include <tune/core/peripherals/EnvironmentState.hpp>
#include <tune/store/RuntimeHistory.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <tune/utils/tupleHelper.hpp>

using namespace tune;
using namespace tune::internal::core::peripherals;
using Arr3u = std::array<uint32_t, 3>;
using TestConfig = config::Config<Arr3u::value_type, 3>;
using TestRecord = config::ConfigRecord<TestConfig>;

TEST_CASE("[TunerConfigQueue] basic behaviour with ConfigRecords", "") {
  ConfigQueue<TestRecord> queue;
  REQUIRE(queue.empty());

  TestConfig c1(Arr3u{1, 2, 3});
  TestConfig c2(Arr3u{4, 5, 6});
  TestRecord r1(c1);
  TestRecord r2(c2);

  queue.try_insert(r1);
  queue.try_insert(r2);

  REQUIRE(queue.size() == 2);
  REQUIRE_FALSE(queue.empty());

  auto recOpt = queue.getConfigFromQueue();
  REQUIRE(recOpt.has_value());

  auto &rec = recOpt->get();
  REQUIRE((rec.m_config == c1 || rec.m_config == c2));
  REQUIRE_FALSE(rec.state == internal::config::ConfigState::Retired);
}

struct DummyRecord {
  uint32_t index;
  internal::config::ConfigState state =
      internal::config::ConfigState::Initialized;
  uint32_t warm_up_runs = 1;
};

TEST_CASE("[TunerConfigQueue] removes fullFlag records automatically", "")

{
  internal::core::peripherals::ConfigQueue<TestRecord> queue;
  TestConfig c1(Arr3u{1, 1, 1});
  TestConfig c2(Arr3u{2, 2, 2});

  TestRecord r1(c1);
  TestRecord r2(c2);

  queue.try_insert(r1);
  queue.try_insert(r2);
  r1.state = internal::config::ConfigState::Retired;
  r2.state = internal::config::ConfigState::Retired;
  REQUIRE(queue.size() == 2);

  auto recOpt = queue.getConfigFromQueue();
  REQUIRE(!recOpt.has_value());
  for (int i = 0; i < 1000; ++i)
    queue.getConfigFromQueue();

  REQUIRE(queue.empty()); // r1 and r2 indirectly removed
}

TEST_CASE("[TunerConfigQueue] random access and cleanup", "") {
  internal::core::peripherals::ConfigQueue<DummyRecord> queue;

  std::vector<DummyRecord> records;
  // queue takes std::ref so you must ensure no reallocation of entries takes
  // place in the tuner this is done by taking only entries that are part active
  // History which are guarenteed to stay at the same address
  records.reserve(10);
  for (unsigned int i = 0; i < 10; ++i) {
    DummyRecord record{i, internal::config::ConfigState::Initialized};
    records.emplace_back(record);
    queue.try_insert(records.back());
  }
  REQUIRE(queue.size() == 10);
  // Mark some configs as full and ensure they're removed
  records[3].state = internal::config::ConfigState::Retired;
  records[5].state = internal::config::ConfigState::Retired;
  records[7].state = internal::config::ConfigState::Retired;

  // Trigger cleanup
  for (int i = 0; i < 1000; ++i)
    queue.getConfigFromQueue();
  REQUIRE(queue.size() == 7);
};

TEST_CASE("[TunerConfigQueue] increase test coverage over additional helper "
          "function and branches",
          "") {
  using sizes = std::tuple<
      std::integral_constant<uint32_t, 2>, std::integral_constant<uint32_t, 1>,
      std::integral_constant<uint32_t, 5>, std::integral_constant<uint32_t, 10>,
      std::integral_constant<uint32_t, 20>,
      std::integral_constant<uint32_t, 50>>;
  using consecutiveRuns = std::tuple<std::integral_constant<uint32_t, 2>,
                                     std::integral_constant<uint32_t, 3>,
                                     std::integral_constant<uint32_t, 5>>;
  using combinations =
      alpaka::meta::CartesianProduct<std::tuple, sizes, consecutiveRuns>;
  meta::for_each_enumerate(combinations{}, [&]<std::size_t I, typename T0>(T0) {
    using SizeT = std::tuple_element_t<0, T0>;
    using RunsT = std::tuple_element_t<1, T0>;
    auto q = ConfigQueue<TestRecord, SizeT::value, RunsT::value>{};
    TestConfig headCfg(Arr3u{111, 111, 111});
    std::vector freshConfigContainer(
        3, TestRecord{TestConfig{Arr3u{111, 111, 111}}});
    auto &headRec = freshConfigContainer[I];

    // Reinsert a fresh HEAD and ensure it’s picked first.
    q.try_insertAdjustHead(headRec);
    CHECK(q.size() >= 1);

    // First get must pick HEAD; first reuse from lastIndex -> config still
    // uninitialized
    auto p1 = q.getConfigFromQueue();
    REQUIRE(p1.has_value());
    auto &first = p1->get();

    CHECK(&first == &headRec);
    CHECK(first.state == internal::config::ConfigState::Uninitialized);
    CHECK(first.warm_up_runs == 0u);

    // Now check reuse window equals maxConsecutiveRuns for this queue
    uint32_t observedReuses = 1u;

    for (;;) {
      auto pn = q.getConfigFromQueue();
      REQUIRE(pn.has_value());

      if (&pn->get() == &headRec) {
        observedReuses++;
        CHECK(observedReuses <= q.m_maxConsecutiveRuns);

        if (observedReuses == q.m_maxConsecutiveRuns)
          break;
      } else {
        CHECK(q.m_maxConsecutiveRuns == 1u);
        break;
      }
    }

    auto pAfterLimit = q.getConfigFromQueue();
    REQUIRE(pAfterLimit.has_value());
    auto &recAfter = pAfterLimit->get();

    if (&recAfter == &headRec) {
      CHECK(recAfter.state == internal::config::ConfigState::WarmUp);
      CHECK(recAfter.warm_up_runs == 0u);
    } else {
      CHECK((recAfter.state == internal::config::ConfigState::WarmUp ||
             recAfter.state == internal::config::ConfigState::InProcess));
    }
  });
}
