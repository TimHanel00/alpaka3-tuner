# Copyright 2026 Tim Hanel
# SPDX-License-Identifier: MPL-2.0

# Compile the README code itself and keep the copyable standalone source in sync.
function(alpakatune_write_readme_example outputFile)
    set(readmeFile "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../README.md")
    set(exampleFile "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../docs/examples/readme/main.cpp")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${readmeFile}" "${exampleFile}")

    file(READ "${readmeFile}" readme)
    string(REGEX MATCH "```cpp\n([^`]+)\n```" readmeExample "${readme}")
    if(NOT readmeExample)
        message(FATAL_ERROR "README.md must contain the complete C++ example in a cpp code block")
    endif()
    set(exampleSource "${CMAKE_MATCH_1}\n")

    file(READ "${exampleFile}" standaloneSource)
    string(
        REGEX REPLACE
        "^// Copyright[^\n]*\n// SPDX-License-Identifier: MPL-2\\.0\n"
        ""
        standaloneSource
        "${standaloneSource}"
    )
    if(NOT exampleSource STREQUAL standaloneSource)
        message(FATAL_ERROR "The README C++ example and docs/examples/readme/main.cpp must match exactly")
    endif()

    file(CONFIGURE OUTPUT "${outputFile}" CONTENT "${exampleSource}" @ONLY)
endfunction()
