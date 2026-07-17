if(CMAKE_CUDA_COMPILER)
    find_package(CUDAToolkit)
endif()

alpakatune_add_example(vendorApi src/main.cpp)
target_include_directories(alpakaTune_vendorApi PRIVATE src)
if(CUDAToolkit_FOUND)
    target_link_libraries(alpakaTune_vendorApi PRIVATE CUDA::cudart_static)
endif()
