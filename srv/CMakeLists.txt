cmake_minimum_required(VERSION 3.9)
project(Graph_Algo)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA)

if(UNIX)
    add_library(srv_UNIX_shm
            UNIX_shm.h
            UNIX_shm.cpp)

    add_library(srv_UNIX_msg
            UNIX_msg.h
            UNIX_msg.cpp)

    add_library(srv_UtilServer
                UtilServer.h
                UtilServer.cpp
                impl/UtilServer_impl.cpp)

    target_link_libraries(srv_UtilServer
            srv_UNIX_msg
            srv_UNIX_shm
            core_Graph
            core_GraphUtil)

    add_library(srv_UtilClient
            UtilClient.h
            UtilClient.cpp
            impl/UtilClient_impl.cpp)

    target_link_libraries(srv_UtilClient
            srv_UNIX_msg
            srv_UNIX_shm
            core_Graph
            core_GraphUtil)

ENDIF(UNIX)