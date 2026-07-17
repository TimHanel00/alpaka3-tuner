alpakatune_add_example(alpaka-ls src/ls.cpp)
target_link_libraries(alpakaTune_alpaka-ls PRIVATE ${CMAKE_DL_LIBS})

foreach(api Host Cuda Hip OneApi)
    string(TOLOWER ${api} api_lower)
    if(TARGET alpaka::${api_lower})
        add_library(alpakaTune_alpaka-ls_${api_lower} SHARED src/libls.cpp)
        target_link_libraries(alpakaTune_alpaka-ls_${api_lower} PRIVATE alpakaTune::alpakaTune alpaka::${api_lower})
        target_compile_definitions(alpakaTune_alpaka-ls_${api_lower} PRIVATE ALPAKA_USE_API=${api})
        set_target_properties(
            alpakaTune_alpaka-ls_${api_lower}
            PROPERTIES OUTPUT_NAME alpaka-ls_${api_lower} LINKER_LANGUAGE CXX FOLDER example
        )
        alpaka_finalize(alpakaTune_alpaka-ls_${api_lower})
        add_dependencies(alpakaTune_alpaka-ls alpakaTune_alpaka-ls_${api_lower})
    endif()
endforeach()
