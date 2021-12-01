cmake_minimum_required(VERSION 3.9)
project(Graph_Algo)

find_package(MPI REQUIRED)

include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA)

add_executable(core_GraphUtilTest
        GraphUtilTest.cpp)

target_link_libraries(core_GraphUtilTest
        core_Graph
        core_GraphUtil
        algo_BellmanFord)

add_executable(algo_BellmanFordTest
        BellmanFordTest.cpp)

target_link_libraries(algo_BellmanFordTest
        algo_BellmanFord
        core_Graph
        core_MessageSet
        core_GraphUtil)
		
add_executable(algo_LabelPropagationTest
		LabelPropagationTest.cpp)

target_link_libraries(algo_LabelPropagationTest
        algo_LabelPropagation
        core_Graph
        core_MessageSet
        core_GraphUtil)

add_executable(algo_ConnectedComponentTest
        ConnectedComponentTest.cpp)

target_link_libraries(algo_ConnectedComponentTest
        algo_ConnectedComponent
        core_Graph
        core_MessageSet
        core_GraphUtil)

add_executable(algo_CubeAggTest
        CubeAggTest.cpp)

target_link_libraries(algo_CubeAggTest
        algo_CubeAgg
        core_Graph
        core_MessageSet
        core_GraphUtil
         pthread)

add_executable(core_GraphAccessTest
        GraphAccessTest.cpp)

target_link_libraries(core_GraphAccessTest
        core_Graph
        core_MessageSet
        core_GraphUtil)

if(UNIX)
    add_executable(srv_Test_srv
            SRVTest_srv.cpp)

    target_link_libraries(srv_Test_srv
            srv_UNIX_shm
            srv_UNIX_msg
            srv_UtilServer
            srv_UtilClient)

    add_executable(srv_Test_cli
            SRVTest_cli.cpp)

    target_link_libraries(srv_Test_cli
            srv_UNIX_shm
            srv_UNIX_msg
            srv_UtilServer
            srv_UtilClient)

    add_executable(srv_UtilServerTest_BellmanFord
            UtilServerTest_BellmanFord.cpp)

    target_link_libraries(srv_UtilServerTest_BellmanFord
            srv_UNIX_shm
            srv_UNIX_msg
            srv_UtilServer
            algo_BellmanFord
            algo_LabelPropagation
            algo_CubeAgg
            pthread)

    add_executable(srv_UtilServerTest_LabelPropagation
            UtilServerTest_LabelPropagation.cpp)

    target_link_libraries(srv_UtilServerTest_LabelPropagation
            srv_UNIX_shm
            srv_UNIX_msg
            srv_UtilServer
            algo_BellmanFord
            algo_LabelPropagation
            algo_CubeAgg
            pthread)

   add_executable(srv_UtilServerTest_CubeAgg
           UtilServerTest_CubeAgg.cpp)

   target_link_libraries(srv_UtilServerTest_CubeAgg
           srv_UNIX_shm
           srv_UNIX_msg
           srv_UtilServer
           algo_BellmanFord
           algo_LabelPropagation
           algo_CubeAgg
           pthread)

    add_executable(srv_UtilClientTest
            UtilClientTest.cpp)

    target_link_libraries(srv_UtilClientTest
            pthread
            srv_UNIX_shm
            srv_UNIX_msg
            srv_UtilClient)

   add_executable(srv_UtilClientTest_CubeAgg
           UtilClientTest_CubeAgg.cpp)

   target_link_libraries(srv_UtilClientTest_CubeAgg
           srv_UNIX_shm
           srv_UNIX_msg
           srv_UtilClient
           algo_BellmanFord
           algo_LabelPropagation
           algo_CubeAgg
           pthread
            ${MPI_LIBRARIES})

ENDIF(UNIX)