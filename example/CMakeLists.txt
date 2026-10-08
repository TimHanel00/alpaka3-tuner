# Copyright 2026 Tim Hanel
# SPDX-License-Identifier: MPL-2.0

macro(alpakaTune_add_example target)
    add_executable(alpakaTune_${target} ${ARGN})
    target_link_libraries(alpakaTune_${target} PRIVATE alpakaTune::alpakaTune)
    target_include_directories(alpakaTune_${target} PRIVATE ${PROJECT_SOURCE_DIR}/example)
    target_compile_features(alpakaTune_${target} PRIVATE cxx_std_20)
    set_target_properties(alpakaTune_${target} PROPERTIES FOLDER example)
    alpaka_finalize(alpakaTune_${target})
endmacro()

function(alpakaTune_add_example_smoke_test target)
    add_test(NAME alpakaTune_${target} COMMAND alpakaTune_${target} ${ARGN})
    set_tests_properties(
        alpakaTune_${target}
        PROPERTIES ENVIRONMENT "ALPAKA_TUNE_CONFIG=${PROJECT_SOURCE_DIR}/tests/adaptive-smoke.yaml"
    )
endfunction()

add_subdirectory(boundaryIter)
add_subdirectory(grayScale)
add_subdirectory(heatEquation2D)
add_subdirectory(matrixMultiplication)
add_subdirectory(matmul)
add_subdirectory(nBody)
add_subdirectory(scan)
add_subdirectory(vectorAdd)
add_subdirectory(tuneTheTuner)
add_subdirectory(tutorial)
add_subdirectory(alpaka-ls)
add_subdirectory(randomInit)
add_subdirectory(postEvaluation)
add_subdirectory(vendorApi)
add_subdirectory(adaptivePi)

if(alpakaTune_DEP_METRICS)
    foreach(_example IN ITEMS vectorAdd heatEquation2D matrixMultiplication)
        if(_example STREQUAL "vectorAdd")
            set(_arguments -n 16384 -r 2)
        elseif(_example STREQUAL "heatEquation2D")
            set(_arguments
                -n
                64
                -t
                20
                -d
                0.0001
            )
        else()
            set(_arguments
                -m
                32
                -n
                32
                -k
                32
                -r
                2
            )
        endif()
        add_test(
            NAME alpakaTune_${_example}_metrics
            COMMAND
                alpakaTune_${_example} ${_arguments} --objective elapsed-time --backend host:cpu --executor CpuSerial
        )
        set_tests_properties(
            alpakaTune_${_example}_metrics
            PROPERTIES ENVIRONMENT "ALPAKA_TUNE_CONFIG=${PROJECT_SOURCE_DIR}/tests/adaptive-smoke.yaml" TIMEOUT 120
        )
    endforeach()
endif()
