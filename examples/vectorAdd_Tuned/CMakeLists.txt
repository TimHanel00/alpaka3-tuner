#
# Copyright 2023-2025 Benjamin Worpitz, Jan Stephan, Tim Hanel
# SPDX-License-Identifier: ISC
#

################################################################################
# Required CMake version.

cmake_minimum_required(VERSION 3.25)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Project.

set(_TARGET_NAME vectorAdd_Tuned)
add_executable(${_TARGET_NAME} src/vectorAdd_Tuned.cpp)
target_link_libraries(${_TARGET_NAME} PUBLIC alpakaTune::alpakaTune)
alpaka_finalize(${_TARGET_NAME})
set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER example)

add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME})
