// Copyright 2026 Tim Hanel
// SPDX-License-Identifier: MPL-2.0
#include "AdaptivePi.hpp"
#include <alpakaTune/BackendSelection.hpp>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  try {
    if (!alpakaTune::consumeBackendOptions(argc, argv))
      return 1;
    double budget = 0.01;
    double maximumError = 1.0e-3;
    auto objective = adaptivePi::Objective::accuracy;
    for (int i = 1; i < argc; ++i) {
      std::string const argument{argv[i]};
      if (argument == "--help") {
        std::cout
            << "Usage: adaptivePi [--backend api:deviceKind] [--executor name] "
               "[--objective accuracy|runtime] "
               "[--runtime-budget-ms 10] [--max-error 1e-3]\n"
               "accuracy minimizes Pi error with a runtime limit; runtime "
               "minimizes elapsed time with an accuracy limit.\n";
        return 0;
      }
      if (argument == "--objective" && i + 1 < argc) {
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
            "Usage: adaptivePi [--backend api:deviceKind] [--executor name] "
            "[--runtime-budget-ms 10] [--objective "
            "accuracy|runtime] [--max-error 1e-3]"};
    }
    bool executed = false;
    auto const status = alpaka::onHost::executeForEach(
        [&](alpaka::concepts::BackendSpec auto const &backend) {
          if (!alpakaTune::backendSelected(backend))
            return 0;
          auto selector = alpaka::onHost::makeDeviceSelector(
              alpaka::onHost::DeviceSpec{backend});
          if (!selector.isAvailable())
            return 0;
          executed = true;
          auto const result = adaptivePi::optimize(
              selector.makeDevice(0u), alpaka::getExecutor(backend), budget,
              objective, maximumError);
          auto const deviceSpec = alpaka::onHost::DeviceSpec{backend};
          std::cout << "backend=" << deviceSpec.getApi().getName() << ":"
                    << deviceSpec.getDeviceKind().getName() << " executor="
                    << alpaka::onHost::demangledName(
                           alpaka::getExecutor(backend))
                    << '\n';
          adaptivePi::require(
              result.feasible,
              "No configuration met the selected post-evaluation constraint");
          std::cout << std::setprecision(12) << "pi=" << result.winner.estimate
                    << " objective="
                    << (objective == adaptivePi::Objective::accuracy
                            ? "accuracy"
                            : "runtime")
                    << " best_metric=" << result.bestMetric
                    << " error=" << result.bestError
                    << " elapsed_ms=" << result.winner.seconds * 1000.0
                    << "\nmaximumSplits=" << result.parameters[0u]
                    << " pointsPerTile=" << result.parameters[1u]
                    << "\ncandidates=" << result.candidateCount
                    << " constraint_rejected=" << result.rejectedCount << '\n';
          return 0;
        },
        alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs,
                                    alpaka::exec::enabledExecutors));
    adaptivePi::require(executed,
                        "No selected backend has an available device");
    return status;
  } catch (std::exception const &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
