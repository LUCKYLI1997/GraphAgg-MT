cmake_minimum_required(VERSION 3.9)
project(Graph_Algo)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA)

add_library(algo_StronglyConnectedComponent
        StronglyConnectedComponent.h
        StronglyConnectedComponent.cpp
        StronglyConnectedComponent_impl.cpp)

target_link_libraries(algo_StronglyConnectedComponent
        core_Graph
        core_GraphUtil
        core_MessageSet)

if(false)
    if(CUDA_FOUND)
        CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; ${ARCH_FLAGS}; -std=c++14;)

        cuda_add_library(algo_StronglyConnectedComponentGPU_kernel STATIC
                kernel_src/StronglyConnectedComponentGPU_kernel.h
                kernel_src/StronglyConnectedComponentGPU_kernel.cu)

        add_library(algo_StronglyConnectedComponentGPU
                StronglyConnectedComponentGPU.h
                StronglyConnectedComponentGPU.cpp
                StronglyConnectedComponentGPU_impl.cpp)

        target_link_libraries(algo_StronglyConnectedComponentGPU
                algo_StronglyConnectedComponent
                algo_StronglyConnectedComponentGPU_kernel)

    endif(CUDA_FOUND)
endif()