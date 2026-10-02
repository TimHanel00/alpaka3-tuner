# Copyright 2026 Tim Hanel
# SPDX-License-Identifier: MPL-2.0
"""Offline CMake ownership regression with intentionally different Alpaka pins."""

import pathlib
import subprocess
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(*args):
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(result.stdout)


def main():
    with tempfile.TemporaryDirectory(prefix="alpakaTune-dependencies-") as temporary:
        directory = pathlib.Path(temporary)
        for owner, version in (("standalone", 1), ("metrics", 2)):
            source = directory / owner
            source.mkdir()
            (source / "owner.hpp").write_text(f"#define ALPAKA_OWNER {version}\n")
            (source / "CMakeLists.txt").write_text(
                f"""cmake_minimum_required(VERSION 3.25)
project({owner} LANGUAGES CXX)
add_library(alpaka INTERFACE)
add_library(alpaka::alpaka ALIAS alpaka)
target_include_directories(alpaka INTERFACE "${{CMAKE_CURRENT_SOURCE_DIR}}")
set_property(TARGET alpaka PROPERTY TEST_OWNER "{owner}")
"""
            )
        # The fake pinned metrics project owns a different Alpaka source and pin.
        metrics = directory / "metrics_project"
        metrics.mkdir()
        (metrics / "CMakeLists.txt").write_text(
            """cmake_minimum_required(VERSION 3.25)
project(metrics_project LANGUAGES CXX)
set(alpakaMetrics_ALPAKA_REVISION metrics-pin CACHE STRING "")
if(NOT alpakaMetrics_ALPAKA_REVISION STREQUAL "metrics-pin")
  message(FATAL_ERROR "Stale revision overrode metrics ownership")
endif()
if(FETCHCONTENT_SOURCE_DIR_ALPAKA3)
  message(FATAL_ERROR "Standalone source override leaked into metrics")
endif()
if(NOT FETCHCONTENT_TRY_FIND_PACKAGE_MODE STREQUAL "NEVER")
  message(FATAL_ERROR "Pinned metrics must not be replaced by automatic package discovery")
endif()
if(NOT FETCHCONTENT_BASE_DIR MATCHES "/_deps/metrics$")
  message(FATAL_ERROR "Metrics dependencies must have a separate directory")
endif()
add_subdirectory("../metrics" "${CMAKE_CURRENT_BINARY_DIR}/alpaka3")
add_library(metrics INTERFACE)
add_library(alpakaMetrics::alpakaMetrics ALIAS metrics)
target_link_libraries(metrics INTERFACE alpaka::alpaka)
"""
        )
        project = directory / "project"
        project.mkdir()
        (project / "main.cpp.in").write_text(
            '#include <owner.hpp>\nstatic_assert(ALPAKA_OWNER == @expected@);\nint main() {}\n'
        )
        (project / "CMakeLists.txt").write_text(
            f"""cmake_minimum_required(VERSION 3.25)
project(switching LANGUAGES CXX)
option(alpakaTune_DEP_METRICS "" OFF)
include("{ROOT / 'cmake/Dependencies.cmake'}")
get_target_property(owner alpaka TEST_OWNER)
if(alpakaTune_DEP_METRICS)
  set(expected 2)
  if(NOT owner STREQUAL "metrics")
    message(FATAL_ERROR "Metrics did not own Alpaka")
  endif()
else()
  set(expected 1)
  if(TARGET alpakaMetrics::alpakaMetrics OR NOT owner STREQUAL "standalone")
    message(FATAL_ERROR "Disabled metrics was still active")
  endif()
endif()
configure_file(main.cpp.in main.cpp @ONLY)
add_executable(consumer "${{CMAKE_CURRENT_BINARY_DIR}}/main.cpp")
target_link_libraries(consumer PRIVATE alpaka::alpaka)
"""
        )
        build = directory / "build"
        for enabled in ("OFF", "ON", "OFF", "ON"):
            run(
                "cmake", "-S", str(project), "-B", str(build), "-G", "Ninja",
                f"-DalpakaTune_DEP_METRICS={enabled}",
                f"-DFETCHCONTENT_SOURCE_DIR_ALPAKA3={directory / 'standalone'}",
                f"-DFETCHCONTENT_SOURCE_DIR_ALPAKAMETRICS={metrics}",
                "-DalpakaMetrics_ALPAKA_REVISION=stale-fallback-pin",
                "-DFETCHCONTENT_TRY_FIND_PACKAGE_MODE=ALWAYS",
            )
            run("cmake", "--build", str(build))
            run(str(build / "consumer"))
    print("OFF -> ON -> OFF -> ON rebuilt with the selected Alpaka owner")


if __name__ == "__main__":
    main()
