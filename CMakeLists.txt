# Copyright 2025 Tim Hanel
# SPDX-License-Identifier: MPL-2.0
cmake_minimum_required(VERSION 3.25)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(alpakaTune)
if(NOT TARGET alpakaTune::alpakaTune)
    #load alpaka cmake
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/alpaka)
    ## load .cmake files
    file(GLOB _cmake_files "${CMAKE_CURRENT_SOURCE_DIR}/cmake/*.cmake")

    ## add alpaka + tuner under common alpakaTune::alpakaTune interface -- this means we do by default only use the file hierarchy (no alpaka install)
    add_library(alpakaTune INTERFACE)
    add_library(alpakaTune::alpakaTune ALIAS alpakaTune)
    target_link_libraries(alpakaTune INTERFACE alpaka::alpaka)
    target_include_directories(alpakaTune INTERFACE ${PROJECT_SOURCE_DIR}/include)
    include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/alpaka/include)
    foreach(_cmake_file ${_cmake_files})
        include(${_cmake_file})
    endforeach()
    #for language specifications (CUDA/HIP) -- resets some variables for internal alpaka_finalize compatability
    alpaka_fetchcontent_finalize()
endif()
option(alpakaTune_HEADERCHECKS "Enable header checks (for files specific to this project)" OFF)
if(alpakaTune_HEADERCHECKS)
    add_subdirectory(headerCheck)
endif()
option(alpakaTune_TESTING "Enable testing for alpaka3-tuner" OFF)
if(alpakaTune_TESTING)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()

option(alpakaTune_EXAMPLES "Enable examples for alpaka3-tuner" OFF)
if(alpakaTune_EXAMPLES)
    add_subdirectory(examples)
endif()
