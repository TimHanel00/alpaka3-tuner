#
# Copyright 2014-2023 Benjamin Worpitz, Axel Huebl, Jan Stephan, Rene Widera
#
# This file is part of PIConGPU.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

cmake_minimum_required(VERSION 3.25)
project(alpaka3Tuner)

set(_TARGET_NAME "headerCheckTest")

################################################################################
# Add subdirectories.
################################################################################

# alpaka
if(NOT TARGET alpakaTuner::alpakaTuner)
    # hide alpaka_HEADERCHECKS because it can be OFF since this test is build directly.
    mark_as_advanced(FORCE alpaka_HEADERCHECKS)
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/.." "${CMAKE_BINARY_DIR}/tuner")
endif()

################################################################################
# Enable CTests
################################################################################

enable_testing()

################################################################################
# Directory of this file.
################################################################################
set(ALPAKATUNER_ROOT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../include/tuner)

# Normalize the path (e.g. remove ../)
get_filename_component(ALPAKATUNER_ROOT_INCLUDE_DIR "${ALPAKA_ROOT_INCLUDE_DIR}" ABSOLUTE)

#---------------------------------------------------------------------------
# Create source files.
set(ALPAKATUNER_SUFFIXED_INCLUDE_DIR "${ALPAKA_ROOT_INCLUDE_DIR}")
append_recursive_files("${ALPAKATUNER_SUFFIXED_INCLUDE_DIR}" "hpp" "ALPAKA_FILES_HEADER")

target_compile_definitions(${_TARGET_NAME} PRIVATE "ALPAKA_NO_TPP_INCLUDE")
target_link_libraries(${_TARGET_NAME} PRIVATE alpaka::alpaka)

set_target_properties(headerCheckTest PROPERTIES FOLDER "headerCheck")

add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME})
