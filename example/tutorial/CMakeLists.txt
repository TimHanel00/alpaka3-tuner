foreach(
    source
    src/00_enumerate.cpp
    src/01_queue.cpp
    src/03_memory.cpp
    src/04_views.cpp
    src/05_kernel.cpp
    src/06_cudaLikeKernel.cpp
    src/07_chunkedData.cpp
)
    get_filename_component(name ${source} NAME_WE)
    alpakatune_add_example(${name} ${source})
    target_include_directories(alpakaTune_${name} PRIVATE src)
endforeach()
