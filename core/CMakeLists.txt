cmake_minimum_required(VERSION 3.9)
project(Graph_Algo)

set(CMAKE_CXX_STANDARD 14)

add_library(core_AbstractGraph
        AbstractGraph.h
        AbstractGraph.cpp)

add_library(core_Graph
        Graph.h
        Graph.cpp
        impl/Graph_impl.cpp)

target_link_libraries(core_Graph
        core_AbstractGraph)

add_library(core_GraphUtil
        GraphUtil.h
        GraphUtil.cpp
        impl/GraphUtil_impl.cpp)

add_library(core_MessageSet
        MessageSet.h
        MessageSet.cpp
        impl/MessageSet_impl.cpp)
