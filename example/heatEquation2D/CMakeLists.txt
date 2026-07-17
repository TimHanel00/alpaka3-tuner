find_package(PNGwriter 0.7.0 CONFIG)

alpakatune_add_example(heatEquation2D src/heatEquation2D.cpp)
target_include_directories(alpakaTune_heatEquation2D PRIVATE src)
if(PNGwriter_FOUND)
    target_link_libraries(alpakaTune_heatEquation2D PRIVATE PNGwriter::PNGwriter)
    target_compile_definitions(alpakaTune_heatEquation2D PRIVATE PNGWRITER_ENABLED)
endif()
