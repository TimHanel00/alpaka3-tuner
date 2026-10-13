# Copyright 2026 Tim Hanel
# SPDX-License-Identifier: MPL-2.0
cmake_minimum_required(VERSION 3.25)
project(readmeExample LANGUAGES CXX)

find_package(alpakaTune CONFIG QUIET)
if(NOT alpakaTune_FOUND)
    include(FetchContent)
    FetchContent_Declare(alpakaTune SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../..")
    FetchContent_MakeAvailable(alpakaTune)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/ReadmeExample.cmake")
set(readmeSource "${CMAKE_CURRENT_BINARY_DIR}/readme.cpp")
alpakatune_write_readme_example("${readmeSource}")
add_executable(readme_example "${readmeSource}")
target_link_libraries(readme_example PRIVATE alpakaTune::alpakaTune)
alpaka_finalize(readme_example)
