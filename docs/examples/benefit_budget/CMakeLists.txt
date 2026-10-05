# Copyright 2026 Tim Hanel
# SPDX-License-Identifier: MPL-2.0
cmake_minimum_required(VERSION 3.25)
project(benefitBudgetExample LANGUAGES CXX)

# Prefer an installed package when CMAKE_PREFIX_PATH points to one.
find_package(alpakaTune CONFIG QUIET)
if(NOT alpakaTune_FOUND)
    include(FetchContent)
    FetchContent_Declare(alpakaTune SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../..")
    FetchContent_MakeAvailable(alpakaTune)
endif()

add_executable(benefit_budget main.cpp)
target_link_libraries(benefit_budget PRIVATE alpakaTune::alpakaTune)
alpaka_finalize(benefit_budget)
