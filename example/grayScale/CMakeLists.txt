find_package(PNGwriter 0.7.0 CONFIG)

alpakatune_add_example(grayScale src/grayScale.cpp)
if(PNGwriter_FOUND)
    target_link_libraries(alpakaTune_grayScale PRIVATE PNGwriter::PNGwriter)
    target_compile_definitions(alpakaTune_grayScale PRIVATE PNGWRITER_ENABLED)
endif()
