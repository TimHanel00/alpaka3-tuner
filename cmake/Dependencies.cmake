# Copyright 2026 Tim Hanel
# SPDX-License-Identifier: MPL-2.0
include(FetchContent)

# Function scope keeps the metrics-owned FetchContent directory and source
# selection from leaking into unrelated dependencies. Targets remain visible.
function(alpakaTune_make_accelerator_dependencies)
    if(alpakaTune_DEP_METRICS)
        set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE NEVER)
        set(FETCHCONTENT_BASE_DIR "${CMAKE_BINARY_DIR}/_deps/metrics")
        # A prior standalone configure may have used a local Alpaka override.
        # Metrics owns its revision, including when switching an existing build.
        set(FETCHCONTENT_SOURCE_DIR_ALPAKA3 "")
        unset(alpakaMetrics_ALPAKA_REVISION CACHE)
        unset(alpakaMetrics_ALPAKA_REVISION)
        FetchContent_Declare(
            alpakaMetrics
            GIT_REPOSITORY https://github.com/TimHanel00/alpakaMetrics.git
            GIT_TAG b2f3fbf5086f875a1fea41dabe198df8bf76f7f1
            GIT_SHALLOW FALSE
        )
        FetchContent_MakeAvailable(alpakaMetrics)
    elseif(NOT TARGET alpaka::alpaka)
        FetchContent_Declare(
            alpaka3
            GIT_REPOSITORY https://github.com/alpaka-group/alpaka3.git
            GIT_TAG b7d339d07056a9a2a6c4051cc3927157bc5f0d51
            # A shallow clone cannot reliably resolve a historical pin.
            GIT_SHALLOW FALSE
        )
        FetchContent_MakeAvailable(alpaka3)
    endif()
endfunction()

alpakatune_make_accelerator_dependencies()
