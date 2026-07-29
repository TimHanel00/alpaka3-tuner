add_executable(alpakaTune_tuner_tests tuner.cpp)
target_link_libraries(alpakaTune_tuner_tests PRIVATE alpakaTune::alpakaTune)
alpaka_finalize(alpakaTune_tuner_tests)
add_test(NAME alpakaTune_tuner_tests COMMAND alpakaTune_tuner_tests)

add_executable(alpakaTune_active_recommendation_tests active_recommendation.cpp)
target_link_libraries(alpakaTune_active_recommendation_tests PRIVATE alpakaTune::alpakaTune Catch2::Catch2WithMain)
alpaka_finalize(alpakaTune_active_recommendation_tests)
add_test(NAME alpakaTune_active_recommendation_tests COMMAND alpakaTune_active_recommendation_tests)

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
