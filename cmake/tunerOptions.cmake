# Copyright 2025 Tim Hanel
# SPDX-License-Identifier: MPL-2.0
# ==========================================================
# === STRATEGY SELECTION ===
# ==========================================================
# Allow user to set strategies using cmake options
set(alpakaTune_Strategy
    "randomExplore"
    CACHE STRING
    "Select the tuning strategy that is used during the tuning process."
)
set_property(CACHE alpakaTune_Strategy PROPERTY STRINGS "randomExplore, exhaustive,randomSample")
# Strategy selection macro
if(alpakaTune_Strategy STREQUAL "randomExplore")
    set(alpakaTune_STRATEGY_MACRO strategy_randomSearch)
elseif(alpakaTune_Strategy STREQUAL "exhaustive")
    set(alpakaTune_STRATEGY_MACRO strategy_exhaustiveSearch)
elseif(alpakaTune_Strategy STREQUAL "randomSample")
    set(alpakaTune_STRATEGY_MACRO strategy_randomSample)
else()
    message(FATAL_ERROR "Invalid DTuner_Strategy: ${alpakaTune_Strategy}")
endif()
# Boolean option (ON/OFF in ccmake)
option(
    alpakaTune_FAST_CONFIG_EVAL
    "Allow prematurely skipping of parameter configs in case of the performance is significantly worse than the best performing configuration."
    OFF
)

# Numeric cache variables (editable in ccmake)
set(alpakaTune_MAX_CONFIG_WINDOW
    50
    CACHE STRING
    "Maximum number of parameter configs that are executed in interleaved fashion."
)

set(alpakaTune_MAX_CONSECUTIVE_RUNS
    3
    CACHE STRING
    "Maximum number of consecutive executions of one particular parameter config."
)

set(alpakaTune_WARMUP_RUNS 1 CACHE STRING "Number of executions of a parameter config before taking a measurement.")
target_compile_definitions(
    alpakaTune
    INTERFACE
        ALLOW_PREMATURE_CONFIG_SKIP=$<BOOL:${ALLOW_PREMATURE_CONFIG_SKIP}>
        MAX_QUEUE_SIZE=${alpakaTune_MAX_CONFIG_WINDOW}
        MAX_CONSECUTIVE_RUNS=${alpakaTune_MAX_CONSECUTIVE_RUNS}
        WARMUP_RUNS=${alpakaTune_WARMUP_RUNS}
        ${alpakaTune_STRATEGY_MACRO}
)
