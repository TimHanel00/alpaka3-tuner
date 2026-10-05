add_executable(alpakaTune_tuner_tests tuner.cpp)
target_link_libraries(alpakaTune_tuner_tests PRIVATE alpakaTune::alpakaTune)
alpaka_finalize(alpakaTune_tuner_tests)
add_test(NAME alpakaTune_tuner_tests COMMAND alpakaTune_tuner_tests)

add_executable(alpakaTune_active_recommendation_tests active_recommendation.cpp)
target_link_libraries(alpakaTune_active_recommendation_tests PRIVATE alpakaTune::alpakaTune Catch2::Catch2WithMain)
alpaka_finalize(alpakaTune_active_recommendation_tests)
add_test(NAME alpakaTune_active_recommendation_tests COMMAND alpakaTune_active_recommendation_tests)

add_executable(alpakaTune_user_invalidation_tests user_invalidation.cpp)
target_link_libraries(alpakaTune_user_invalidation_tests PRIVATE alpakaTune::alpakaTune Catch2::Catch2WithMain)
alpaka_finalize(alpakaTune_user_invalidation_tests)
add_test(NAME alpakaTune_user_invalidation_tests COMMAND alpakaTune_user_invalidation_tests)

add_executable(alpakaTune_custom_metric_tests custom_metric.cpp)
target_link_libraries(alpakaTune_custom_metric_tests PRIVATE alpakaTune::alpakaTune Catch2::Catch2WithMain)
alpaka_finalize(alpakaTune_custom_metric_tests)
add_test(NAME alpakaTune_custom_metric_tests COMMAND alpakaTune_custom_metric_tests)

add_executable(alpakaTune_frame_spec_tuning_tests frame_spec_tuning.cpp)
target_link_libraries(alpakaTune_frame_spec_tuning_tests PRIVATE alpakaTune::alpakaTune)
alpaka_finalize(alpakaTune_frame_spec_tuning_tests)
add_test(NAME alpakaTune_frame_spec_tuning_tests COMMAND alpakaTune_frame_spec_tuning_tests)

add_executable(alpakaTune_candidate_queue_tests candidate_queue.cpp)
target_link_libraries(alpakaTune_candidate_queue_tests PRIVATE alpakaTune::alpakaTune)
alpaka_finalize(alpakaTune_candidate_queue_tests)
add_test(NAME alpakaTune_candidate_queue_tests COMMAND alpakaTune_candidate_queue_tests)

add_executable(alpakaTune_strategy_tests strategy.cpp)
target_link_libraries(alpakaTune_strategy_tests PRIVATE alpakaTune::alpakaTune)
alpaka_finalize(alpakaTune_strategy_tests)
add_test(NAME alpakaTune_strategy_tests COMMAND alpakaTune_strategy_tests)

add_executable(alpakaTune_learned_hybrid_tests learned_hybrid_strategy.cpp)
target_link_libraries(alpakaTune_learned_hybrid_tests PRIVATE alpakaTune::alpakaTune)
alpaka_finalize(alpakaTune_learned_hybrid_tests)
add_test(NAME alpakaTune_learned_hybrid_tests COMMAND alpakaTune_learned_hybrid_tests)

add_executable(alpakaTune_runtime_history_tests runtime_history.cpp)
target_link_libraries(alpakaTune_runtime_history_tests PRIVATE alpakaTune::alpakaTune)
alpaka_finalize(alpakaTune_runtime_history_tests)
add_test(NAME alpakaTune_runtime_history_tests COMMAND alpakaTune_runtime_history_tests)

add_executable(alpakaTune_history_tests history.cpp)
target_link_libraries(alpakaTune_history_tests PRIVATE alpakaTune::alpakaTune)
alpaka_finalize(alpakaTune_history_tests)
add_test(NAME alpakaTune_history_tests COMMAND alpakaTune_history_tests)

add_executable(alpakaTune_tune_the_tuner_tests tune_the_tuner.cpp)
target_link_libraries(alpakaTune_tune_the_tuner_tests PRIVATE alpakaTune::alpakaTune Catch2::Catch2WithMain)
alpaka_finalize(alpakaTune_tune_the_tuner_tests)
add_test(NAME alpakaTune_tune_the_tuner_tests COMMAND alpakaTune_tune_the_tuner_tests)

if(alpakaTune_ENABLE_PERSISTENCE)
    add_executable(alpakaTune_persistence_shutdown_tests persistence_shutdown.cpp)
    target_link_libraries(alpakaTune_persistence_shutdown_tests PRIVATE alpakaTune::alpakaTune)
    alpaka_finalize(alpakaTune_persistence_shutdown_tests)
    set(_alpakaTune_partial_history "${CMAKE_CURRENT_BINARY_DIR}/partial-history.json")
    set(_alpakaTune_completed_history "${CMAKE_CURRENT_BINARY_DIR}/completed-history.json")
    set(_alpakaTune_write_only_history "${CMAKE_CURRENT_BINARY_DIR}/write-only-history.json")
    set(_alpakaTune_read_only_history "${CMAKE_CURRENT_BINARY_DIR}/read-only-history.json")
    add_test(
        NAME alpakaTune_persistence_shutdown_stage
        COMMAND
            alpakaTune_persistence_shutdown_tests stage "${_alpakaTune_partial_history}"
            "${_alpakaTune_completed_history}" "${_alpakaTune_write_only_history}" "${_alpakaTune_read_only_history}"
    )
    set_tests_properties(
        alpakaTune_persistence_shutdown_stage
        PROPERTIES FIXTURES_SETUP alpakaTune_persistence_shutdown
    )
    add_test(
        NAME alpakaTune_persistence_shutdown_verify
        COMMAND
            alpakaTune_persistence_shutdown_tests verify "${_alpakaTune_partial_history}"
            "${_alpakaTune_completed_history}" "${_alpakaTune_write_only_history}" "${_alpakaTune_read_only_history}"
    )
    set_tests_properties(
        alpakaTune_persistence_shutdown_verify
        PROPERTIES FIXTURES_REQUIRED alpakaTune_persistence_shutdown
    )
endif()

add_executable(alpakaTune_workflow_tests workflows.cpp)
target_link_libraries(alpakaTune_workflow_tests PRIVATE alpakaTune::alpakaTune)
alpaka_finalize(alpakaTune_workflow_tests)
foreach(_workflow IN ITEMS pipeline stencil quality)
    add_test(NAME alpakaTune_workflow_${_workflow} COMMAND alpakaTune_workflow_tests ${_workflow})
    set_tests_properties(alpakaTune_workflow_${_workflow} PROPERTIES LABELS integration TIMEOUT 120)
endforeach()
if(alpakaTune_ENABLE_PERSISTENCE)
    set(_workflow_history "${CMAKE_CURRENT_BINARY_DIR}/workflow-history")
    add_test(
        NAME alpakaTune_workflow_persistence_stage
        COMMAND alpakaTune_workflow_tests pipeline stage "${_workflow_history}"
    )
    set_tests_properties(
        alpakaTune_workflow_persistence_stage
        PROPERTIES FIXTURES_SETUP alpakaTune_workflow_history LABELS integration TIMEOUT 120
    )
    add_test(
        NAME alpakaTune_workflow_persistence_replay
        COMMAND alpakaTune_workflow_tests pipeline replay "${_workflow_history}"
    )
    set_tests_properties(
        alpakaTune_workflow_persistence_replay
        PROPERTIES FIXTURES_REQUIRED alpakaTune_workflow_history LABELS integration TIMEOUT 120
    )
endif()

add_executable(alpakaTune_adaptive_pi_tests adaptive_pi.cpp)
target_link_libraries(alpakaTune_adaptive_pi_tests PRIVATE alpakaTune::alpakaTune Catch2::Catch2WithMain)
target_include_directories(
    alpakaTune_adaptive_pi_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/example/adaptivePi/src ${CMAKE_CURRENT_SOURCE_DIR}
)
alpaka_finalize(alpakaTune_adaptive_pi_tests)
add_test(NAME alpakaTune_adaptive_pi_tests COMMAND alpakaTune_adaptive_pi_tests)
# Exhaustive device calculations and leaf inspection are substantially slower
# under sanitizers; the runtime objective is checked independently of this limit.
set_tests_properties(alpakaTune_adaptive_pi_tests PROPERTIES LABELS integration TIMEOUT 600)

# Compile the exact source shown by the Sphinx tutorial in both candidate modes.
foreach(_tutorial_variant IN ITEMS runtime compile_time)
    set(_tutorial_target alpakaTune_tutorial_${_tutorial_variant})
    add_executable(${_tutorial_target} ${PROJECT_SOURCE_DIR}/docs/examples/first_tuner/main.cpp)
    target_link_libraries(${_tutorial_target} PRIVATE alpakaTune::alpakaTune)
    if(_tutorial_variant STREQUAL "compile_time")
        target_compile_definitions(${_tutorial_target} PRIVATE ALPAKA_TUNE_TUTORIAL_COMPILE_TIME=1)
    endif()
    alpaka_finalize(${_tutorial_target})
    add_test(NAME ${_tutorial_target} COMMAND ${_tutorial_target})
    set_tests_properties(${_tutorial_target} PROPERTIES PASS_REGULAR_EXPRESSION "candidates=3.*completed=1.*loaded=0")
    if(alpakaTune_ENABLE_PERSISTENCE)
        set(_tutorial_workdir "${CMAKE_CURRENT_BINARY_DIR}/tutorial-${_tutorial_variant}")
        file(MAKE_DIRECTORY "${_tutorial_workdir}")
        foreach(_tutorial_policy IN ITEMS collect replay adaptive)
            add_test(
                NAME ${_tutorial_target}_${_tutorial_policy}
                COMMAND ${_tutorial_target} ${PROJECT_SOURCE_DIR}/docs/examples/first_tuner/${_tutorial_policy}.yaml
            )
            set_tests_properties(
                ${_tutorial_target}_${_tutorial_policy}
                PROPERTIES WORKING_DIRECTORY "${_tutorial_workdir}"
            )
        endforeach()
        set_tests_properties(
            ${_tutorial_target}_collect
            PROPERTIES FIXTURES_SETUP ${_tutorial_target}_history PASS_REGULAR_EXPRESSION "completed=1.*loaded=0"
        )
        set_tests_properties(
            ${_tutorial_target}_replay
            PROPERTIES FIXTURES_REQUIRED ${_tutorial_target}_history PASS_REGULAR_EXPRESSION "completed=1.*loaded=1"
        )
        set_tests_properties(
            ${_tutorial_target}_adaptive
            PROPERTIES FIXTURES_REQUIRED ${_tutorial_target}_history PASS_REGULAR_EXPRESSION "completed=0.*loaded=1"
        )
    endif()
endforeach()

add_executable(alpakaTune_managed_space_tests managed_space.cpp)
target_link_libraries(alpakaTune_managed_space_tests PRIVATE alpakaTune::alpakaTune Catch2::Catch2WithMain)
alpaka_finalize(alpakaTune_managed_space_tests)
add_test(NAME alpakaTune_managed_space_tests COMMAND alpakaTune_managed_space_tests)
