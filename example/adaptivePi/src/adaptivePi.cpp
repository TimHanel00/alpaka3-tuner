// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0
#include "AdaptivePi.hpp"
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  try {
    bool cuda = false;
    double budget = 0.01;
    double maximumError = 1.0e-3;
    auto objective = adaptivePi::Objective::accuracy;
    for (int i = 1; i < argc; ++i) {
      std::string const argument{argv[i]};
      if (argument == "--help") {
        std::cout
            << "Usage: adaptivePi [--cuda] [--objective accuracy|runtime] "
               "[--runtime-budget-ms 10] [--max-error 1e-3]\n"
               "accuracy minimizes Pi error with a runtime limit; runtime "
               "minimizes elapsed time with an accuracy limit.\n";
        return 0;
      }
      if (argument == "--cuda")
        cuda = true;
      else if (argument == "--objective" && i + 1 < argc) {
        std::string const value{argv[++i]};
        adaptivePi::require(value == "accuracy" || value == "runtime",
                            "Unknown objective");
        objective = value == "accuracy" ? adaptivePi::Objective::accuracy
                                        : adaptivePi::Objective::runtime;
      } else if (argument == "--max-error" && i + 1 < argc)
        maximumError = std::stod(argv[++i]);
      else if (argument == "--runtime-budget-ms" && i + 1 < argc)
        budget = std::stod(argv[++i]) / 1000.0;
      else
        throw std::runtime_error{
            "Usage: adaptivePi [--cuda] [--runtime-budget-ms 10] [--objective "
            "accuracy|runtime] [--max-error 1e-3]"};
    }
    auto result = adaptivePi::Optimization{};
    if (cuda) {
#if ALPAKA_LANG_CUDA && defined(ALPAKA_CMAKE_TARGET_CUDA)
      auto selector =
          alpaka::onHost::makeDeviceSelector(alpaka::onHost::DeviceSpec{
              alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu});
      adaptivePi::require(selector.isAvailable(), "CUDA device unavailable");
      result =
          adaptivePi::optimize(selector.makeDevice(0u), alpaka::exec::gpuCuda,
                               budget, objective, maximumError);
#else
      throw std::runtime_error{"CUDA support was not compiled in"};
#endif
    } else {
      auto selector =
          alpaka::onHost::makeDeviceSelector(alpaka::onHost::DeviceSpec{
              alpaka::api::host, alpaka::deviceKind::cpu});
      result =
          adaptivePi::optimize(selector.makeDevice(0u), alpaka::exec::cpuSerial,
                               budget, objective, maximumError);
    }
    adaptivePi::require(
        result.feasible,
        "No configuration met the selected post-evaluation constraint");
    std::cout << std::setprecision(12) << "pi=" << result.winner.estimate
              << " objective="
              << (objective == adaptivePi::Objective::accuracy ? "accuracy"
                                                               : "runtime")
              << " best_metric=" << result.bestMetric
              << " error=" << result.bestError
              << " elapsed_ms=" << result.winner.seconds * 1000.0
              << "\nmaximumSplits=" << result.parameters[0u]
              << " pointsPerTile=" << result.parameters[1u]
              << " numFrames=" << result.parameters[2u]
              << " frameExtent=" << result.parameters[3u]
              << "\ncandidates=" << result.candidateCount
              << " constraint_rejected=" << result.rejectedCount << '\n';
    return 0;
  } catch (std::exception const &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
