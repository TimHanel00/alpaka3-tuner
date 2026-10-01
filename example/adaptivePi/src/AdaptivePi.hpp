// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0
#pragma once

#include <algorithm>
#include <alpaka/alpaka.hpp>
#include <alpakaTune/alpakaTune.hpp>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace adaptivePi {
using Index = alpaka::Vec<std::size_t, 1u>;
inline constexpr auto maximumSplits = ALPAKA_TUNE_TUNABLE("maximumSplits");
inline constexpr auto pointsPerTile = ALPAKA_TUNE_TUNABLE("pointsPerTile");
// These parameters describe the inner calculation's FrameSpec. The outer
// host kernel runs once to orchestrate the complete multi-launch workflow.
inline constexpr auto calculationFrames =
    ALPAKA_TUNE_TUNABLE("calculationFrames");
inline constexpr auto calculationExtent =
    ALPAKA_TUNE_TUNABLE("calculationExtent");

inline void require(bool condition, char const *message) {
  if (!condition)
    throw std::runtime_error{message};
}

/** A square in [0,1] x [0,1]; id identifies its deterministic random stream. */
struct Tile {
  double x{}, y{}, side{1.0};
  std::uint32_t id{1u};
};
enum class Classification : std::uint32_t { inside, outside, boundary };
struct TileResult {
  Classification classification{Classification::boundary};
  std::uint32_t hits{};
};

// Counter-based integer mixing keeps test results reproducible independently
// of worker order, executor, and frame geometry. Each tile has its own stream.
ALPAKA_FN_HOST_ACC inline auto randomBits(std::uint32_t value)
    -> std::uint32_t {
  value ^= value >> 16u;
  value *= 0x7feb352du;
  value ^= value >> 15u;
  value *= 0x846ca68bu;
  return value ^ (value >> 16u);
}
ALPAKA_FN_HOST_ACC inline auto inside(double x, double y) -> bool {
  return x * x + y * y < 1.0;
}

/** Classify independent tiles, sampling boundary tiles only at the final depth.
 * All four corners inside a convex circle prove the complete square is inside.
 * A boundary tile otherwise remains undecided until refinement or sampling.
 */
struct ClassifyTiles {
  ALPAKA_FN_ACC void operator()(alpaka::onAcc::concepts::Acc auto const &acc,
                                alpaka::concepts::IMdSpan auto tiles,
                                alpaka::concepts::IMdSpan auto results,
                                Index extent, std::uint32_t points,
                                bool finalDepth) const {
    for (auto index :
         alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid,
                                   alpaka::IdxRange{extent})) {
      auto const tile = tiles[index];
      auto result = TileResult{};
      if (inside(tile.x, tile.y) && inside(tile.x + tile.side, tile.y) &&
          inside(tile.x, tile.y + tile.side) &&
          inside(tile.x + tile.side, tile.y + tile.side)) {
        result.classification = Classification::inside;
      } else if (!inside(tile.x, tile.y)) {
        // In the first quadrant the lower-left corner is the nearest point.
        // Four outside corners alone would not prove that a tile is outside.
        result.classification = Classification::outside;
      } else if (finalDepth) {
        for (std::uint32_t sample = 0u; sample < points; ++sample) {
          auto const counter = tile.id * 0x9e3779b9u + sample * 2u;
          double const x =
              tile.x +
              tile.side * ((static_cast<double>(randomBits(counter)) + 0.5) /
                           4294967296.0);
          double const y =
              tile.y +
              tile.side *
                  ((static_cast<double>(randomBits(counter + 1u)) + 0.5) /
                   4294967296.0);
          result.hits += inside(x, y) ? 1u : 0u;
        }
      }
      results[index] = result;
    }
  }
};

struct Calculation {
  double estimate{}, lowerBound{}, upperBound{}, seconds{};
  double partitionArea{};
  std::size_t insideTiles{}, outsideTiles{}, splitTiles{}, sampledTiles{},
      randomPoints{};
  std::uint32_t deepestSample{};
  std::array<std::uint32_t, 4u> parameters{};
};

template <alpaka::onHost::concepts::Device Device,
          alpaka::concepts::Executor Executor>
class Calculator {
public:
  Calculator(Device device, Executor executor)
      : m_device(std::move(device)), m_executor(executor),
        m_queue(m_device.makeQueue()),
        m_tiles(alpaka::onHost::allocHost<Tile>(Index{capacity})),
        m_results(alpaka::onHost::allocHost<TileResult>(Index{capacity})),
        m_deviceTiles(alpaka::onHost::allocLike(m_device, m_tiles)),
        m_deviceResults(alpaka::onHost::allocLike(m_device, m_results)) {}

  ~Calculator() {
    // Finish asynchronous work before the owning buffers are released.
    alpaka::onHost::wait(m_queue);
  }

  auto calculate(std::uint32_t splits, std::uint32_t points,
                 std::uint32_t frames, std::uint32_t frameExtent)
      -> Calculation {
    require(splits <= 9u && points > 0u && frames > 0u && frameExtent > 0u,
            "Invalid adaptive Pi calculation parameters");
    // Measure one complete calculation: root setup, all parallel phases,
    // frontier construction, transfers, and reduction. Allocation of reusable
    // buffers belongs to Calculator construction and is outside this budget.
    auto const start = std::chrono::steady_clock::now();
    auto frontier = std::vector<Tile>{Tile{}};
    auto calculation = Calculation{};
    double insideArea = 0.0, outsideArea = 0.0, boundaryArea = 0.0,
           sampledArea = 0.0;
    for (std::uint32_t depth = 0u; !frontier.empty(); ++depth) {
      require(frontier.size() <= capacity,
              "Pi frontier exceeded buffer capacity");
      auto const extent = Index{frontier.size()};
      std::copy(frontier.begin(), frontier.end(), m_tiles.data());
      alpaka::onHost::memcpy(m_queue, m_deviceTiles, m_tiles, extent);
      auto const frame = alpaka::onHost::FrameSpec{
          Index{frames}, Index{frameExtent}, m_executor};
      m_queue.enqueue(frame, alpaka::KernelBundle{
                                 ClassifyTiles{}, m_deviceTiles.getMdSpan(),
                                 m_deviceResults.getMdSpan(), extent, points,
                                 depth == splits});
      alpaka::onHost::memcpy(m_queue, m_results, m_deviceResults, extent);
      alpaka::onHost::wait(m_queue);
      // Retired inside/outside tiles never re-enter this frontier. Only
      // unresolved boundary tiles create children, each with one quarter area.
      auto next = std::vector<Tile>{};
      for (std::size_t i = 0u; i < frontier.size(); ++i) {
        auto const tile = frontier[i];
        auto const result = m_results[i];
        double const area = tile.side * tile.side;
        if (result.classification == Classification::inside) {
          insideArea += area;
          ++calculation.insideTiles;
        } else if (result.classification == Classification::outside) {
          outsideArea += area;
          ++calculation.outsideTiles;
        } else if (depth == splits) {
          require(result.hits <= points, "Invalid Monte Carlo hit count");
          boundaryArea += area;
          sampledArea += area * result.hits / points;
          ++calculation.sampledTiles;
          calculation.randomPoints += points;
          calculation.deepestSample = depth;
        } else {
          require(result.hits == 0u,
                  "Monte Carlo sampling occurred before final depth");
          ++calculation.splitTiles;
          double const half = tile.side / 2.0;
          for (std::uint32_t child = 0u; child < 4u; ++child)
            next.push_back(Tile{tile.x + (child % 2u) * half,
                                tile.y + (child / 2u) * half, half,
                                tile.id * 4u + child});
        }
      }
      frontier = std::move(next);
    }
    // Sum exact interior area and Monte Carlo boundary area. The factor four
    // converts quarter-circle area into Pi; leaf areas also provide geometric
    // lower/upper bounds independent of the random hit counts.
    calculation.estimate = 4.0 * (insideArea + sampledArea);
    calculation.lowerBound = 4.0 * insideArea;
    calculation.upperBound = 4.0 * (insideArea + boundaryArea);
    calculation.partitionArea = insideArea + outsideArea + boundaryArea;
    calculation.parameters = {splits, points, frames, frameExtent};
    calculation.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    last = calculation;
    return calculation;
  }
  Calculation last{};
  std::exception_ptr failure{};

private:
  static constexpr std::size_t capacity = 16384u;
  Device m_device;
  Executor m_executor;
  decltype(std::declval<Device &>().makeQueue()) m_queue;
  decltype(alpaka::onHost::allocHost<Tile>(Index{capacity})) m_tiles;
  decltype(alpaka::onHost::allocHost<TileResult>(Index{capacity})) m_results;
  decltype(alpaka::onHost::allocLike(m_device, m_tiles)) m_deviceTiles;
  decltype(alpaka::onHost::allocLike(m_device, m_results)) m_deviceResults;
};

// This outer kernel is exclusively a host orchestration entry point. Keeping
// it host-only also allows NVCC to register the inner CUDA tile kernel
// normally.
struct RunCalculation {
  template <typename Engine>
  ALPAKA_FN_HOST void operator()(alpaka::onAcc::concepts::Acc auto const &,
                                 Engine *engine, std::uint32_t splits,
                                 std::uint32_t points, std::uint32_t frames,
                                 std::uint32_t extent) const {
    try {
      engine->calculate(splits, points, frames, extent);
    } catch (...) {
      // Propagate workflow failures after the host queue has completed.
      engine->failure = std::current_exception();
    }
  }
};

inline auto tuningConfig() -> alpakaTune::TunerConfig {
  auto config = alpakaTune::TunerConfig{};
  config.mode = alpakaTune::TuningMode::onlineFixed;
  config.replayFastPath = true;
  config.strategy = alpakaTune::StrategyKind::exhaustive;
  config.queue.reset();
  config.runsPerCandidate = 1u;
  config.minimumRunsPerCandidate = 1u;
  config.maximumExecutions = 1024u;
  config.maximumRetiredConfigurations.reset();
  config.mannWhitneyEarlyStop = false;
  config.history = {.read = false, .write = false};
  config.completeHistory = {.read = false, .write = false};
  return config;
}

struct Optimization {
  bool feasible{};
  Calculation winner{};
  std::size_t candidateCount{}, rejectedCount{}, replayCount{};
  double bestError{std::numeric_limits<double>::infinity()};
  double bestMetric{std::numeric_limits<double>::infinity()};
  std::array<std::uint32_t, 4u> parameters{};
};

inline void validate(Calculation const &result, std::uint32_t splits,
                     std::uint32_t points) {
  if (result.partitionArea != 1.0)
    throw std::runtime_error{
        "Pi tiles do not partition the unit square: area=" +
        std::to_string(result.partitionArea) +
        " splits=" + std::to_string(result.parameters[0u]) +
        " points=" + std::to_string(result.parameters[1u]) +
        " inside=" + std::to_string(result.insideTiles) +
        " outside=" + std::to_string(result.outsideTiles) +
        " sampled=" + std::to_string(result.sampledTiles)};
  require(result.lowerBound <= std::numbers::pi &&
              std::numbers::pi <= result.upperBound,
          "Geometric Pi bounds exclude the reference");
  require(std::isfinite(result.estimate) &&
              result.lowerBound <= result.estimate &&
              result.estimate <= result.upperBound,
          "Pi estimate violates tile bounds");
  require(result.insideTiles > 0u && result.outsideTiles > 0u &&
              result.splitTiles > 0u && result.sampledTiles > 0u,
          "Pi calculation did not exercise all tile classifications");
  require(result.deepestSample == splits &&
              result.randomPoints == result.sampledTiles * points,
          "Pi calculation used the wrong sampling depth or point budget");
}

/** Select which quantity is minimized and which is a post-run constraint. */
enum class Objective { accuracy, runtime };
struct AccuracyObjective {
  static constexpr bool custom = true;
};
struct RuntimeObjective {
  static constexpr bool custom = false;
};

// One calculation engine and one tuning loop serve both objectives. The policy
// only selects the queue instrumentation, tuner metric, and validity decision.
template <typename Policy>
auto optimizeImpl(alpaka::onHost::concepts::Device auto device,
                  alpaka::concepts::Executor auto executor,
                  double budgetSeconds, double maximumError, Policy)
    -> Optimization {
  require(std::isfinite(budgetSeconds) && budgetSeconds >= 0.0,
          "Invalid runtime budget");
  require(std::isfinite(maximumError) && maximumError >= 0.0,
          "Invalid accuracy limit");
  auto calculator = Calculator{device, executor};
  auto selector = alpaka::onHost::makeDeviceSelector(
      alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu});
  auto host = selector.makeDevice(0u);
  auto queue = [&] {
    if constexpr (Policy::custom)
      return alpakaTune::makeQueue(host, alpaka::queueKind::nonBlocking,
                                   alpakaTune::timing::disabled);
    else
      return alpakaTune::makeQueue(host, alpaka::queueKind::nonBlocking,
                                   alpakaTune::timing::enabled);
  }();
  // A single host invocation orchestrates the entire calculation. Its inner
  // queues may run on a GPU; calculate() waits for every phase. Built-in timing
  // therefore measures the complete workflow, including transfers/refinement.
  auto const frame =
      alpaka::onHost::FrameSpec{Index{1u}, Index{1u}, alpaka::exec::cpuSerial};
  // 4 * 5 * 4 * 4 = 320 independent configurations. Inner frame geometry is
  // passed explicitly so it never changes the once-only outer invocation.
  auto const tunables = alpakaTune::TunableBundle{
      maximumSplits(alpakaTune::RVals<std::uint32_t>::list(2u, 4u, 6u, 8u)),
      pointsPerTile(
          alpakaTune::RVals<std::uint32_t>::list(32u, 64u, 128u, 256u, 512u)),
      calculationFrames(alpakaTune::RVals<std::uint32_t>::list(1u, 2u, 4u, 8u)),
      calculationExtent(
          alpakaTune::RVals<std::uint32_t>::list(32u, 64u, 128u, 256u))};
  auto tuner = [&] {
    if constexpr (Policy::custom)
      return alpakaTune::makeTuner(
          tuningConfig(), tunables, host,
          alpakaTune::customMetric("absolute_pi_error"), device, executor,
          "adaptive-pi");
    else
      return alpakaTune::makeTuner(tuningConfig(), tunables, host, device,
                                   executor, "adaptive-pi");
  }();
  auto const bundle = alpaka::KernelBundle{
      RunCalculation{}, &calculator,       maximumSplits,
      pointsPerTile,    calculationFrames, calculationExtent};
  auto outcome = Optimization{};
  outcome.candidateCount = tuner.info().candidateCount;
  std::vector<double> metrics(outcome.candidateCount,
                              std::numeric_limits<double>::infinity());
  for (std::size_t launch = 0u; launch < 512u; ++launch) {
    bool const replay = tuner.completed();
    try {
      tuner.enqueue(queue, frame, bundle);
    } catch (std::exception const &) {
      if (tuner.completed() &&
          tuner.completionReason() ==
              alpakaTune::TunerCompletionReason::noValidConfiguration)
        break;
      throw;
    }
    alpaka::onHost::wait(queue);
    if (calculator.failure)
      std::rethrow_exception(calculator.failure);
    auto const candidate = tuner.lastCandidateIndex();
    auto const &result = calculator.last;
    validate(result, result.parameters[0u], result.parameters[1u]);
    double const error = std::abs(result.estimate - std::numbers::pi);
    bool valid = false;
    if constexpr (Policy::custom) {
      // This is the application-provided objective. Submit it before the next
      // enqueue; it is independent of the post-evaluation runtime constraint.
      tuner.provideMetric(error);
      valid = result.seconds <= budgetSeconds;
      require(!tuner.lastConfig().runtimeSeconds,
              "Pi custom metric used kernel timing");
      if (valid && !replay)
        metrics[candidate] = error;
    } else {
      // No provideMetric() call: this tuner uses the default timing policy.
      // Reject a measured candidate when the actual Pi result is inaccurate.
      valid = error <= maximumError;
      if (valid && !replay) {
        require(tuner.lastConfig().runtimeSeconds.has_value(),
                "Pi runtime measurement missing");
        metrics[candidate] = *tuner.lastConfig().runtimeSeconds;
      }
    }
    if (!valid) {
      tuner.lastConfig().valid = false;
      metrics[candidate] = std::numeric_limits<double>::infinity();
      outcome.replayCount = 0u;
      outcome.feasible = false;
    }
    outcome.bestMetric = *std::min_element(metrics.begin(), metrics.end());
    if (replay && valid) {
      require(!tuner.lastConfig().measured, "Pi replay added a measurement");
      auto const samples = tuner.candidateMetricSamples(candidate);
      require(samples.size() == 1u && samples.front() == outcome.bestMetric,
              "Pi replay did not minimize the feasible objective");
      outcome.winner = result;
      outcome.parameters = result.parameters;
      outcome.bestError = error;
      outcome.feasible = true;
      if (++outcome.replayCount == 4u)
        break;
    }
  }
  outcome.rejectedCount = tuner.info().userInvalidatedCandidateCount;
  require(tuner.completed(), "Pi tuning exceeded its bounded launch count");
  require(tuner.info().missingMetricCandidateCount == 0u,
          "Pi metric was not submitted");
  if (!outcome.feasible) {
    require(outcome.rejectedCount == outcome.candidateCount,
            "Pi search failed without rejecting all candidates");
    require(tuner.completionReason() ==
                alpakaTune::TunerCompletionReason::noValidConfiguration,
            "Pi search reported the wrong no-winner reason");
  }
  return outcome;
}

auto optimize(alpaka::onHost::concepts::Device auto device,
              alpaka::concepts::Executor auto executor,
              double budgetSeconds = 0.01,
              Objective objective = Objective::accuracy,
              double maximumError = 1.0e-3) -> Optimization {
  if (objective == Objective::accuracy)
    return optimizeImpl(device, executor, budgetSeconds, maximumError,
                        AccuracyObjective{});
  return optimizeImpl(device, executor, budgetSeconds, maximumError,
                      RuntimeObjective{});
}
} // namespace adaptivePi
