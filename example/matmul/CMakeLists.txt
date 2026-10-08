# Copyright 2026 Tim Hanel
# SPDX-License-Identifier: MPL-2.0

option(alpakaTune_MATMUL_CUBLAS "Compare the matmul example with strict FP32 cuBLAS" ${alpaka_DEP_CUDA})

add_library(alpakaTune_matmulExample INTERFACE)
add_library(alpakaTune::matmulExample ALIAS alpakaTune_matmulExample)
target_link_libraries(alpakaTune_matmulExample INTERFACE alpaka::alpaka)
target_compile_features(alpakaTune_matmulExample INTERFACE cxx_std_20)
target_include_directories(alpakaTune_matmulExample INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

alpakatune_add_example(matmul main.cpp)
target_link_libraries(alpakaTune_matmul PRIVATE alpakaTune::matmulExample)
target_compile_definitions(alpakaTune_matmul PRIVATE ALPAKA_TUNE_MATMUL_CUBLAS=$<BOOL:${alpakaTune_MATMUL_CUBLAS}>)
if(alpakaTune_MATMUL_CUBLAS)
    if(NOT alpaka_DEP_CUDA)
        message(FATAL_ERROR "alpakaTune_MATMUL_CUBLAS requires alpaka_DEP_CUDA")
    endif()
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(alpakaTune_matmul PRIVATE CUDA::cublas)
endif()

if(alpakaTune_BUILD_TESTING)
    if(alpaka_EXEC_CpuSerial)
        add_test(
            NAME alpakaTune_matmul_correctness
            COMMAND alpakaTune_matmul --self-test --backend host:cpu --executor cpuSerial
        )
        set_tests_properties(alpakaTune_matmul_correctness PROPERTIES TIMEOUT 120)
    endif()
    if(alpaka_DEP_CUDA AND alpaka_EXEC_GpuCuda)
        add_test(
            NAME alpakaTune_matmul_cuda_correctness
            COMMAND alpakaTune_matmul --self-test --backend cuda:nvidiaGpu --executor gpuCuda
        )
        set_tests_properties(alpakaTune_matmul_cuda_correctness PROPERTIES TIMEOUT 120)
    endif()
    if(alpaka_DEP_OMP AND alpaka_EXEC_CpuOmpBlocks)
        add_test(
            NAME alpakaTune_matmul_omp_correctness
            COMMAND alpakaTune_matmul --self-test --backend host:cpu --executor cpuOmpBlocks
        )
        set_tests_properties(
            alpakaTune_matmul_omp_correctness
            PROPERTIES TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=4;OMP_DYNAMIC=FALSE"
        )
    endif()
endif()
