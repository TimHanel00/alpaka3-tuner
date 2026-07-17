find_package(PNGwriter 0.7.0 CONFIG)

alpakatune_add_example(nBody src/nBody.cpp)
target_include_directories(alpakaTune_nBody PRIVATE src)
if(PNGwriter_FOUND)
    target_link_libraries(alpakaTune_nBody PRIVATE PNGwriter::PNGwriter)
    target_compile_definitions(alpakaTune_nBody PRIVATE PNGWRITER_ENABLED)
endif()
