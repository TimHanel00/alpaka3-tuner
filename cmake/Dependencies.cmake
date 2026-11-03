# Copyright 2026 Tim Hanel
# SPDX-License-Identifier: MPL-2.0
include(FetchContent)

# The tuner owns the alpaka revision. alpakaMetrics only fetches alpaka when no
# alpaka::alpaka target exists, so it builds against the same revision.
# Function scope keeps the metrics-owned FetchContent directory and source
# selection from leaking into unrelated dependencies. Targets remain visible.
function(alpakaTune_make_accelerator_dependencies)
    if(NOT TARGET alpaka::alpaka)
        FetchContent_Declare(
            alpaka3
            GIT_REPOSITORY https://github.com/alpaka-group/alpaka.git
            GIT_TAG
                eb677e8d7f70d79b35df25d9002f94ba669b6f44 # 3.0.0
            # A shallow clone cannot reliably resolve a historical pin.
            GIT_SHALLOW FALSE
        )
        FetchContent_MakeAvailable(alpaka3)
    endif()
    if(alpakaTune_DEP_METRICS)
        set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE NEVER)
        set(FETCHCONTENT_BASE_DIR "${CMAKE_BINARY_DIR}/_deps/metrics")
        FetchContent_Declare(
            alpakaMetrics
            GIT_REPOSITORY https://github.com/TimHanel00/alpakaMetrics.git
            GIT_TAG b2f3fbf5086f875a1fea41dabe198df8bf76f7f1
            GIT_SHALLOW FALSE
        )
        FetchContent_MakeAvailable(alpakaMetrics)
    endif()
endfunction()

alpakatune_make_accelerator_dependencies()
