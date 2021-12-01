cmake_minimum_required(VERSION 3.9)
project(Graph_Algo)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA)

add_library(algo_LabelPropagation
        LabelPropagation.h
        LabelPropagation.cpp
        LabelPropagation_impl.cpp)

target_link_libraries(algo_LabelPropagation
        core_Graph
        core_GraphUtil
        core_MessageSet)

if(false)
    if(CUDA_FOUND)
        CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; ${ARCH_FLAGS}; -std=c++14;)

        cuda_add_library(algo_LabelPropagationGPU_kernel STATIC
                kernel_src/LabelPropagationGPU_kernel.h
                kernel_src/LabelPropagationGPU_kernel.cu)

        add_library(algo_LabelPropagationGPU
                LabelPropagationGPU.h
                LabelPropagationGPU.cpp)

        target_link_libraries(algo_LabelPropagationGPU
                algo_LabelPropagation
                algo_LabelPropagationGPU_kernel)

    endif(CUDA_FOUND)
endif()