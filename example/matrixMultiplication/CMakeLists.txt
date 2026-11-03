# Copyright 2026 Tim Hanel
# SPDX-License-Identifier: MPL-2.0

alpakatune_add_example(matrixMultiplication src/matrixMultiplication.cpp)

if(alpakaTune_DEP_METRICS)
    target_link_libraries(alpakaTune_matrixMultiplication PRIVATE alpakaTune::metrics)
endif()
