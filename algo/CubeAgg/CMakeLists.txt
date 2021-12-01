cmake_minimum_required(VERSION 3.9)
project(Graph_Algo)

set(CMAKE_CXX_STANDARD 14)


add_library(algo_CubeAgg
        CubeAgg.h
        CubeAgg.cpp)

target_link_libraries(algo_CubeAgg
        core_Graph
        core_GraphUtil
        core_MessageSet)