#
# Copyright 2025 Tim Hanel
#
# This file is part of the dedicated tuning tool for alpaka3
#
#

cmake_minimum_required(VERSION 3.25)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(alpakaTune)
if(NOT TARGET alpaka::alpakaTune)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/alpaka)
    ## load .cmake files
    file(GLOB _cmake_files "${CMAKE_CURRENT_SOURCE_DIR}/cmake/*.cmake")

    ## add alpaka + tuner under common alpaka::alpakaTune interface -- this means we do by default only use the file hierarchy (no alpaka install)
    add_library(alpakaTune INTERFACE)
    add_library(alpaka::alpakaTune ALIAS alpakaTune)
    target_include_directories(alpakaTune INTERFACE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(alpakaTune INTERFACE alpaka::alpaka)
    foreach(_cmake_file ${_cmake_files})
        include(${_cmake_file})
    endforeach()
endif()
option(tuner_HEADERCHECKS "Enable header checks (for files specific to this project)" OFF)
if(tuner_HEADERCHECKS)
    add_subdirectory(headerCheck)
endif()
option(tuner_TESTING "Enable testing for alpaka3-tuner" OFF)
if(tuner_TESTING)
    add_subdirectory(tests)
endif()

option(tuner_EXAMPLES "Enable examples for alpaka3-tuner" OFF)
if(tuner_EXAMPLES)
    add_subdirectory(examples)
endif()
