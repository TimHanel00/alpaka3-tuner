# Copyright 2025 Tim Hanel
# SPDX-License-Identifier: MPL-2.0
function(alpakaTune_finalize target)
    if(NOT TARGET alpaka::alpaka)
        message(
            FATAL_ERROR
            "alpaka::alpaka target not found. "
            "Did you add the alpaka subdirectory or find_package(alpaka) properly?"
        )
    endif()

    target_link_libraries(${target} PUBLIC alpaka::alpaka)

    # Delegate to Alpaka's finalize logic
    alpaka_finalize(${target})
endfunction()
