# Copyright 2026 Tim Hanel
# SPDX-License-Identifier: MPL-2.0
cmake_minimum_required(VERSION 3.25)
project(tunerTutorial LANGUAGES CXX)

# Prefer an installed package when CMAKE_PREFIX_PATH points to one.
find_package(alpakaTune CONFIG QUIET)
if(NOT alpakaTune_FOUND)
    include(FetchContent)
    FetchContent_Declare(alpakaTune SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../..")
    FetchContent_MakeAvailable(alpakaTune)
endif()

option(alpakaTune_TUTORIAL_COMPILE_TIME "Build the CVals version of the tutorial" OFF)
add_executable(first_tuner main.cpp)
target_link_libraries(first_tuner PRIVATE alpakaTune::alpakaTune)
if(alpakaTune_TUTORIAL_COMPILE_TIME)
    target_compile_definitions(first_tuner PRIVATE ALPAKA_TUNE_TUTORIAL_COMPILE_TIME=1)
endif()
alpaka_finalize(first_tuner)
