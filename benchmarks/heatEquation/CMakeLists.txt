# Copyright 2026 Tim Hanel
# SPDX-License-Identifier: MPL-2.0

set(alpakaTune_HEAT_UPSTREAM_SOURCE
    "${alpaka3_SOURCE_DIR}"
    CACHE PATH
    "alpaka3 source providing the original heat-equation baseline"
)
set(alpakaTune_HEAT_SOURCE_REVISION
    "unrecorded"
    CACHE STRING
    "Revision or source snapshot identity for heat-equation measurements"
)
option(alpakaTune_HEAT_CPU_CATALOG "Use CPU tiles for ompBlocks tuning" OFF)
set(alpakaTune_HEAT_WINNERS_HEADER
    "${CMAKE_CURRENT_SOURCE_DIR}/SelectedWinners.hpp"
    CACHE FILEPATH
    "Header with independently measured heat-equation winners"
)
if(NOT EXISTS "${alpakaTune_HEAT_UPSTREAM_SOURCE}/example/heatEquation2D/src/StencilKernel.hpp")
    message(FATAL_ERROR "Set alpakaTune_HEAT_UPSTREAM_SOURCE to the alpaka3 source tree for the heat-equation baseline")
endif()

foreach(mode IN ITEMS search replay)
    if(mode STREQUAL "search")
        set(target alpakaTune_heatEquation)
        set(replay 0)
    else()
        set(target alpakaTune_heatEquation_replay)
        set(replay 1)
    endif()
    add_executable(${target} main.cpp)
    target_link_libraries(${target} PRIVATE alpakaTune::alpakaTune)
    target_include_directories(
        ${target}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} "${alpakaTune_HEAT_UPSTREAM_SOURCE}/example/heatEquation2D/src"
    )
    target_compile_definitions(${target} PRIVATE ALPAKA_TUNE_HEAT_REPLAY=${replay})
    target_compile_definitions(${target} PRIVATE ALPAKA_TUNE_HEAT_CPU_CATALOG=$<BOOL:${alpakaTune_HEAT_CPU_CATALOG}>)
    target_compile_definitions(${target} PRIVATE ALPAKA_TUNE_HEAT_WINNERS_HEADER="${alpakaTune_HEAT_WINNERS_HEADER}")
    target_compile_definitions(${target} PRIVATE ALPAKA_TUNE_HEAT_SOURCE_REVISION="${alpakaTune_HEAT_SOURCE_REVISION}")
    alpaka_finalize(${target})
endforeach()

if(alpakaTune_BUILD_TESTING)
    add_test(
        NAME alpakaTune_heatEquation_correctness
        COMMAND alpakaTune_heatEquation --self-test --backend host:cpu --executor cpuSerial
    )
    set_tests_properties(alpakaTune_heatEquation_correctness PROPERTIES TIMEOUT 120)
endif()
