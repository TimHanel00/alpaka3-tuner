# Copyright 2025 Tim Hanel
# SPDX-License-Identifier: MPL-2.0
cmake_minimum_required(VERSION 3.25)

project(alpakaTune VERSION 3.0.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(alpakaTune_BUILD_TESTING "Build alpakaTune tests" OFF)
option(alpakaTune_BUILD_EXAMPLES "Build alpakaTune examples" OFF)
option(alpakaTune_HEADER_CHECKS "Build public-header self-containment checks" OFF)
option(alpakaTune_ENABLE_PERSISTENCE "Enable JSON persistent histories" ON)

if(NOT TARGET alpaka::alpaka)
    include(FetchContent)
    FetchContent_Declare(
        alpaka3
        GIT_REPOSITORY https://github.com/alpaka-group/alpaka3.git
        GIT_TAG 2c7ec1e2c27a006de6c26f1368c4718f12827727
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(alpaka3)
endif()

if(NOT TARGET yaml-cpp::yaml-cpp)
    include(FetchContent)
    set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_INSTALL ON CACHE BOOL "" FORCE)
    FetchContent_Declare(yaml_cpp GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git GIT_TAG 0.8.0 GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(yaml_cpp)
endif()

if(alpakaTune_ENABLE_PERSISTENCE AND NOT TARGET nlohmann_json::nlohmann_json)
    include(FetchContent)
    set(JSON_Install ON CACHE BOOL "Install nlohmann/json with alpakaTune" FORCE)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.12.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

add_library(alpakaTune INTERFACE)
add_library(alpakaTune::alpakaTune ALIAS alpakaTune)
target_compile_features(alpakaTune INTERFACE cxx_std_20)
target_include_directories(
    alpakaTune
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(alpakaTune INTERFACE alpaka::alpaka)
target_link_libraries(alpakaTune INTERFACE yaml-cpp::yaml-cpp)
target_compile_definitions(
    alpakaTune
    INTERFACE ALPAKA_TUNE_DEFAULT_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/config/alpakaTune.yaml"
)

if(alpakaTune_ENABLE_PERSISTENCE)
    target_link_libraries(alpakaTune INTERFACE nlohmann_json::nlohmann_json)
    target_compile_definitions(alpakaTune INTERFACE ALPAKA_TUNE_HAS_JSON=1)
else()
    target_compile_definitions(alpakaTune INTERFACE ALPAKA_TUNE_HAS_JSON=0)
endif()

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES config/alpakaTune.yaml DESTINATION ${CMAKE_INSTALL_DATADIR}/alpakaTune)
configure_package_config_file(
    cmake/alpakaTuneConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/alpakaTuneConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/alpakaTune
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/alpakaTuneConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/alpakaTuneConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/alpakaTuneConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/alpakaTune
)

if(alpakaTune_HEADER_CHECKS)
    add_subdirectory(headerCheck)
endif()

if(alpakaTune_BUILD_TESTING)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()

if(alpakaTune_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
