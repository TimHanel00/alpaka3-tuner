# Copyright 2026 Tim Hanel
# SPDX-License-Identifier: MPL-2.0

option(alpakaTune_GEMM_VENDOR "Compare GEMM with alpakaVendor" ON)

if(alpakaTune_GEMM_VENDOR AND NOT TARGET alpakaVendor::alpakaVendor)
    include(FetchContent)
    # BLAS is the only vendor module needed here. Backends remain selectable
    # through alpakaVendor's own dependency options.
    set(alpakaV_DEP_FFTW OFF CACHE BOOL "Enable FFTW host backend")
    set(alpakaV_DEP_OPENBLAS OFF CACHE BOOL "Enable OpenBLAS host backend")
    set(alpakaV_DEP_CUBLAS ${alpaka_DEP_CUDA} CACHE BOOL "Enable cuBLAS CUDA backend")
    set(alpakaV_DEP_ROCBLAS ${alpaka_DEP_HIP} CACHE BOOL "Enable rocBLAS HIP backend")
    set(alpakaV_TESTS OFF CACHE BOOL "Build alpakaVendor tests")
    set(alpakaV_DOCS OFF CACHE BOOL "Build alpakaVendor documentation")
    FetchContent_Declare(
        alpakaVendor
        GIT_REPOSITORY https://github.com/psychocoderHPC/alpakaVendor.git
        GIT_TAG ccf128e3c1ff85ae0b1fc5d9b08fd9bc44f02aeb
        GIT_SUBMODULES ""
    )
    FetchContent_MakeAvailable(alpakaVendor)
endif()

add_executable(alpakaTune_gemm main.cpp)
target_link_libraries(alpakaTune_gemm PRIVATE alpakaTune::alpakaTune)
target_include_directories(alpakaTune_gemm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(alpakaTune_gemm PRIVATE ALPAKA_TUNE_GEMM_VENDOR=$<BOOL:${alpakaTune_GEMM_VENDOR}>)
if(alpakaTune_GEMM_VENDOR)
    target_link_libraries(alpakaTune_gemm PRIVATE alpakaVendor::alpakaVendor)
endif()
alpaka_finalize(alpakaTune_gemm)

if(alpakaTune_BUILD_TESTING)
    add_test(
        NAME alpakaTune_gemm_correctness
        COMMAND alpakaTune_gemm --self-test --backend host:cpu --executor cpuSerial
    )
    set_tests_properties(alpakaTune_gemm_correctness PROPERTIES TIMEOUT 120)
endif()
