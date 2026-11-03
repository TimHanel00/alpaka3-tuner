alpakatune_add_example(vectorAdd src/vectorAdd.cpp)
alpakatune_add_example_smoke_test(vectorAdd -n 16384 -r 1)

if(alpakaTune_DEP_METRICS)
    target_link_libraries(alpakaTune_vectorAdd PRIVATE alpakaTune::metrics)
endif()
