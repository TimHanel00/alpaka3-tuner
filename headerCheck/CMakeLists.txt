cmake_minimum_required(VERSION 3.25)
project(TunerHeaderCheck)

set(_TARGET_NAME tunerHeaderCheckTest)

# Location of your headers
set(TUNER_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include/alpakaTune")

# Collect all *.hpp recursively
file(GLOB_RECURSE TUNER_HEADERS RELATIVE "${TUNER_INCLUDE_DIR}" "${TUNER_INCLUDE_DIR}/*.hpp")

# Directory where generated .cpp files go
set(_GENERATED_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(REMOVE_RECURSE "${_GENERATED_SOURCE_DIR}")
file(MAKE_DIRECTORY "${_GENERATED_SOURCE_DIR}")

set(_GENERATED_SOURCES "")

# Create a .cpp for each header that includes it twice
foreach(_H ${TUNER_HEADERS})
    set(_OUT "${_GENERATED_SOURCE_DIR}/${_H}.cpp")

    # Make sure the directory exists
    get_filename_component(_OUT_DIR "${_OUT}" DIRECTORY)
    file(MAKE_DIRECTORY "${_OUT_DIR}")

    file(WRITE "${_OUT}" "#include <alpakaTune/${_H}>\n" "#include <alpakaTune/${_H}>\n")

    list(APPEND _GENERATED_SOURCES "${_OUT}")
endforeach()

# Add a dummy main
set(_MAIN "${_GENERATED_SOURCE_DIR}/main.cpp")
file(WRITE "${_MAIN}" "int main() { return 0; }\n")
list(APPEND _GENERATED_SOURCES "${_MAIN}")

# Build the executable
add_executable(${_TARGET_NAME} ${_GENERATED_SOURCES})
target_link_libraries(tunerHeaderCheckTest PRIVATE alpakaTune::alpakaTune)
alpakatune_finalize(tunerHeaderCheckTest)
# Add include path
target_include_directories(${_TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")

# You can link other libs if needed:
# target_link_libraries(${_TARGET_NAME} PRIVATE alpaka::alpaka)

# Add a CTest entry
enable_testing()
add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME})
