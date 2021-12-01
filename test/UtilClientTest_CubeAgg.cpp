#include "../core/Graph.h"
#include "../core/GraphUtil.h"
#include "../srv/UtilClient.h"
#include "../srv/UNIX_shm.h"
#include "../srv/UNIX_msg.h"

#include "../algo/CubeAgg/CubeAgg.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>

#include <future>
#include <cstring>

#include <mpi.h>

typedef char byte;

template <typename VertexValueType, typename MessageValueType>
void testFut(UtilClient<VertexValueType, MessageValueType>* uc, int vCount, int eCount, VertexValueType* vValues, Vertex* vSet, Edge* eSet, int* initVSet, bool* filteredV, int filteredVCount)
{
    auto tmp_numOfInitV = uc->numOfInitV;
    uc->vCount = std::max(vCount,1);
    uc->eCount = std::max(eCount, 1);
    uc->numOfInitV = 13;

    uc->connect();
    uc->transfer(vValues, vSet, eSet, initVSet, filteredV, filteredVCount);
    uc->request();
    uc->disconnect();

    uc->numOfInitV = tmp_numOfInitV;
}

int main(int argc, char* argv[])
{
    if (argc != 8)
    {
        std::cout << "Usage:" << std::endl << "./UtilClientTest_CubeAgg vCount eCount dCount nodeCount partitionPerNodeCount threadCount cuboidID" << std::endl;
        return 1;
    }
    int vCount = atoi(argv[1]);
    int eCount = atoi(argv[2]);
    int dCount = atoi(argv[3]);
    int nodeCount = atoi(argv[4]);
    int partitionPerNodeCount = atoi(argv[5]);
    int threadCount = atoi(argv[6]);
    int cuboidID = atoi(argv[7]);
    int totalPartitionCount = nodeCount * partitionPerNodeCount;

    int vCount_perPartition = (vCount + totalPartitionCount - 1) / totalPartitionCount;
    int eCount_perPartition = (eCount + totalPartitionCount - 1) / totalPartitionCount;

    //Parameter check
    if (dCount <= 0 || nodeCount <= 0 || partitionPerNodeCount <= 0 || threadCount <= 0)
    {
        std::cout << "Parameter illegal" << std::endl;
        return 3;
    }

    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    
    printf("This is node%d of %d machines, which names %s\n", world_rank, world_size, processor_name);

    //dimension-selection
    
    // --- dCount = 3 ---
    std::vector<std::vector<int>> aggDimension_3 =
    { {1,1,1}, {1,1,0}, {1,0,1}, {0,1,1}, {1,0,0}, {0,1,0}, {0,0,1}, {0,0,0} };
    std::vector<std::string> name_3 =
    { "111","110","101","011","100","010","001","000" };
    std::vector<int> dgList_3 = { -1,0,0,0,0,0,0,0 };
    //std::vector<int> dgList_3 = { -1,0,0,0,0,3,3,3 };
    //                             0,1,2,3,4,5,6,7

    // --- dCount = 4 ---
    std::vector<std::vector<int>> aggDimension_4 =
    { {1,1,1,1},
      {1,1,1,0}, {1,1,0,1}, {1,0,1,1}, {0,1,1,1},
      {1,1,0,0}, {1,0,1,0}, {1,0,0,1}, {0,1,1,0}, {0,1,0,1}, {0,0,1,1},
      {1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,0,1},
      {0,0,0,0}
    };
    std::vector<std::string> name_4 =
    { "1111",
      "1110","1101", "1011", "0111",
      "1100", "1010", "1001", "0110", "0101", "0011",
      "1000", "0100", "0010", "0001",
      "0000"
    };
    std::vector<int> dgList_4 = { -1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
    //std::vector<int> dgList_4 = { -1,0,0,0,0,2,0,2,0,2, 0, 7, 9, 6, 9, 7 };
    //                             0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

    // --- dCount = 5 ---
    std::vector<std::vector<int>> aggDimension_5 =
    { {1,1,1,1,1},
      {1,1,1,1,0},{1,1,1,0,1},{1,1,0,1,1},{1,0,1,1,1},{0,1,1,1,1},
      {1,1,1,0,0},{1,1,0,1,0},{1,1,0,0,1},{1,0,1,1,0},{1,0,1,0,1},{1,0,0,1,1},{0,1,1,1,0},{0,1,1,0,1},{0,1,0,1,1},{0,0,1,1,1},
      {1,1,0,0,0},{1,0,1,0,0},{1,0,0,1,0},{1,0,0,0,1},{0,1,1,0,0},{0,1,0,1,0},{0,1,0,0,1},{0,0,1,1,0},{0,0,1,0,1},{0,0,0,1,1},
      {1,0,0,0,0},{0,1,0,0,0},{0,0,1,0,0},{0,0,0,1,0},{0,0,0,0,1},
      {0,0,0,0,0}
    };
    std::vector<std::string> name_5 =
    { "11111",
      "11110","11101", "11011", "10111", "01111",
      "11100","11010","11001","10110","10101","10011","01110","01101","01011","00111",
      "11000","10100","10010","10001","01100","01010","01001","00110","00101","00011",
      "10000","01000", "00100", "00010", "00001",
      "00000"
    };
    std::vector<int> dgList_5 = { -1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
    //std::vector<int> dgList_5 = { -1,0,0,0,0,0,1,1,0,1, 0, 0, 1, 0, 0, 0, 8, 9, 9, 8, 1, 1, 8, 9,10,11, 8, 8, 9, 9, 8, 8};
    //                             0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31

    // assign dCount
    std::vector<std::vector<int>>& aggDimension = aggDimension_3;
    std::vector<std::string>& name = name_3;
    std::vector<int>& dgList = dgList_3;
    if (dCount == 3)
    {
        aggDimension = aggDimension_3;
        name = name_3;
        dgList = dgList_3;
    }
    else if (dCount == 4)
    {
        aggDimension = aggDimension_4;
        name = name_4;
        dgList = dgList_4;
    }
    else if (dCount == 5)
    {
        aggDimension = aggDimension_5;
        name = name_5;
        dgList = dgList_5;
    }
    else
    {
        std::cout << "illegal dimension count" << std::endl;
        return 5;
    }
    
    int cuboidCount = 0;

    MPI_Barrier(MPI_COMM_WORLD);    //before branch

    //Master Process
    if (world_rank == 0)
    {
        //Open Graph File
        std::ifstream GinVertex("testGraphVertex.txt");
        if (!GinVertex.is_open()) { std::cout << "Error! File testGraphVertex.txt not found!" << std::endl; return 4; }
        std::ifstream GinEdge("testGraphEdge.txt");
        if (!GinEdge.is_open()) { std::cout << "Error! File testGraphEdge.txt not found!" << std::endl; return 4; }

        int tmp;
        GinVertex >> tmp;
        if (vCount != tmp)
        {
            std::cout << "Graph file doesn't match up UtilClient's parameter" << std::endl;
            return 5;
        }
        GinEdge >> tmp;
        if (eCount != tmp)
        {
            std::cout << "Graph file doesn't match up UtilClient's parameter" << std::endl;
            return 5;
        }

        //Buffer between master and workers
        std::vector<std::vector<byte*>> send_buffer_graph_element(nodeCount, std::vector<byte*>(partitionPerNodeCount));    //vSet or eSet
        std::vector<std::vector<byte*>> send_buffer_graph_other(nodeCount, std::vector<byte*>(partitionPerNodeCount));      //attrSet measure or v2v
        std::vector<std::vector<byte*>> recv_buffer_graph_element(nodeCount, std::vector<byte*>(partitionPerNodeCount));    //AggVertex Set or AggEdge Set
        std::vector<std::vector<int*>> send_buffer_element_count(nodeCount, std::vector<int*>(partitionPerNodeCount));      //vCount or eCount
        std::vector<std::vector<int*>> recv_buffer_element_count(nodeCount, std::vector<int*>(partitionPerNodeCount));      //aggCount
        std::vector<std::vector<int*>> recv_buffer_maxTime_compute(nodeCount, std::vector<int*>(partitionPerNodeCount));    //maxTimeCompute
        std::vector<std::vector<int*>> recv_buffer_maxTime_merge(nodeCount, std::vector<int*>(partitionPerNodeCount));      //maxTimeMerge
        std::vector<std::vector<int*>> recv_buffer_mergeCycle(nodeCount, std::vector<int*>(partitionPerNodeCount));         //mergeCycle

        int send_buffer_graph_element_size = std::max(10 * vCount_perPartition * sizeof(Vertex), 10 * eCount_perPartition * sizeof(Edge));
        int recv_buffer_graph_element_size = std::max((unsigned)(10 * sizeof(AggVertex) * vCount_perPartition), (unsigned)(10 * eCount_perPartition * sizeof(AggEdge)));
        int send_buffer_graph_other_size = std::max((unsigned)(10 * sizeof(CubeAggVertexValue) * vCount_perPartition), (unsigned)(10 * (vCount + 1) * sizeof(int)));


        for (int nid = 0; nid < nodeCount; nid++)
        {
            for (int pid = 0; pid < partitionPerNodeCount; pid++)
            {
                send_buffer_graph_element[nid][pid] = new byte[send_buffer_graph_element_size];
                recv_buffer_graph_element[nid][pid] = new byte[recv_buffer_graph_element_size];
                send_buffer_graph_other[nid][pid] = new byte[send_buffer_graph_other_size];
                send_buffer_element_count[nid][pid] = new int;
                recv_buffer_element_count[nid][pid] = new int;
                recv_buffer_maxTime_compute[nid][pid] = new int;
                recv_buffer_maxTime_merge[nid][pid] = new int;
                recv_buffer_mergeCycle[nid][pid] = new int;
            }
        }

        //Create MultiDimensional Graph
        CubeAggGraph test;
        test.vCount = vCount;
        test.vList = std::vector<Vertex>(vCount);
        test.verticesValue = std::vector<CubeAggVertexValue>(vCount);
        for (int i = 0; i < vCount; i++)
        {
            int vertexID;
            std::vector<std::string> d(dCount);
            GinVertex >> vertexID;
            for (int j = 0; j < dCount; j++)
            {
                GinVertex >> test.verticesValue[vertexID].dimension_ptr[j];
            }
            GinVertex >> test.verticesValue[vertexID].measure;
            test.vList[i].vertexID = vertexID;
        }
        for (int i = 0; i < eCount; i++)
        {
            int src, dst;
            double weight;

            GinEdge >> src >> dst >> weight;
            test.eList.emplace_back(src, dst, weight);
            test.eCount++;
            test.vList[src].outDegree++;
            test.vList[dst].inDegree++;
        }
        GinVertex.close();
        GinEdge.close();

        auto testUtilClient = CubeAgg<CubeAggVertexValue, int>(); //Main Worker
        testUtilClient.currentThread = threadCount;
        testUtilClient.currentPartition = totalPartitionCount;
        testUtilClient.dCount = dCount;

        //Test
        std::cout << "Init finished" << std::endl;
        //Test end

        std::vector<CubeAggGraph> cuboidSet(pow(2, dCount));   // Save Cuboids

        int workerCount(0);
        MPI_Comm_size(MPI_COMM_WORLD, &workerCount);
        workerCount--;

        // ApplyD
        while (cuboidCount < pow(2, dCount))
        {
            MPI_Barrier(MPI_COMM_WORLD);    //1-beginning

            cuboidCount++;

            if (cuboidCount != cuboidID)	continue;

            //Test
            std::cout << "Start:" << clock() << std::endl;
            //Test end

            //assign input graph
            CubeAggGraph* gd;
            if (dgList.at(cuboidCount - 1) == -1)
                gd = &test;
            else
                gd = &cuboidSet.at(dgList.at(cuboidCount - 1));

            std::vector<std::vector<AggVertex>> aggVListSet(totalPartitionCount);  // intermediate agg sub graph set
            std::vector<std::vector<AggEdge>> aggEListSet(totalPartitionCount);  // intermediate agg sub graph set

            testUtilClient.AggInit();

            auto subGraphSet = testUtilClient.DivideGraph((*gd), testUtilClient.currentPartition, 
                threadCount, aggDimension.at(cuboidCount - 1));            

            //=== VC begin ===
            
            // transfrom vertex data into send_buffer
            for (int nid = 0; nid < nodeCount; nid++)
            {
                for (int pid = 0; pid < partitionPerNodeCount; pid++)
                {
                    int subg_id = nid * partitionPerNodeCount + pid;
                    //vSet
                    memcpy(send_buffer_graph_element[nid][pid], &subGraphSet[subg_id].vList[0], subGraphSet[subg_id].vCount * sizeof(Vertex));               
                    //vertex imf
                    memcpy(send_buffer_graph_other[nid][pid], &subGraphSet[subg_id].verticesValue[0], subGraphSet[subg_id].vCount * sizeof(CubeAggVertexValue));                    
                    //vCount
                    *send_buffer_element_count[nid][pid] = subGraphSet[subg_id].vCount;
                }
            }

            //send data in send_buffer to each worker
            for (int nid = 0; nid < workerCount; nid++)
            {
                for (int pid = 0; pid < partitionPerNodeCount; pid++) 
                {
                    //vSet: Tag = 0 + 10 * pid
                    MPI_Send(send_buffer_graph_element[nid][pid], send_buffer_graph_element_size, MPI_BYTE, nid + 1, 0 + 10 * pid, MPI_COMM_WORLD);
                    //vertexImf: Tag = 1 + 10 * pid
                    MPI_Send(send_buffer_graph_other[nid][pid], send_buffer_graph_other_size, MPI_BYTE, nid + 1, 1 + 10 * pid, MPI_COMM_WORLD);
                    //control: Tag = 2 + 10 * pid
                    MPI_Send(send_buffer_element_count[nid][pid], 1, MPI_INT, nid + 1, 2 + 10 * pid, MPI_COMM_WORLD);
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);    //2-after sending vertex data

            // receive vertex partitions data from all worker 
            for (int nid = 0; nid < workerCount; nid++)
            {
                for (int pid = 0; pid < partitionPerNodeCount; pid++)
                {
                    //agg vSet: Tag = 0 + pid * 10
                    MPI_Recv(recv_buffer_graph_element[nid][pid], recv_buffer_graph_element_size, MPI_BYTE, nid + 1, 0 + pid * 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    //aggCount: Tag = 1 + pid * 10
                    MPI_Recv(recv_buffer_element_count[nid][pid], 1, MPI_INT, nid + 1, 1 + pid * 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    //maxTimeCompute: Tag = 2 + pid * 10
                    MPI_Recv(recv_buffer_maxTime_compute[nid][pid], 1, MPI_INT, nid + 1, 2 + pid * 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    //maxTimeMerge: Tag = 3 + pid * 10
                    MPI_Recv(recv_buffer_maxTime_merge[nid][pid], 1, MPI_INT, nid + 1, 3 + pid * 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    //mergeCycle: Tag = 4 + pid * 10
                    MPI_Recv(recv_buffer_mergeCycle[nid][pid], 1, MPI_INT, nid + 1, 4 + pid * 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }

            // transform the data into aggVListSet
            for (int nid = 0; nid < nodeCount; nid++)
            {
                for (int pid = 0; pid < partitionPerNodeCount; pid++)
                {
                    //index of subG
                    int subg_id = pid + nid * partitionPerNodeCount;

                    //aggCount
                    int aggCount = *recv_buffer_element_count[nid][pid];                    

                    //agg vSet
                    int AggVertexSize = sizeof(AggVertex);
                    for (int i = 0; i < aggCount; i++)
                    {
                        long long address = (long long)recv_buffer_graph_element[nid][pid] + i * AggVertexSize;
                        AggVertex tmp = *(AggVertex*)address;
                        aggVListSet[subg_id].push_back(tmp);
                    }
                }
            }

            // record the time and mergeCycle
            int maxTimeCompute = 0;
            int maxTimeMerge = 0;
            int mergeCycle = 0;
            for (int nid = 0; nid < nodeCount; nid++)
            {
                for (int pid = 0; pid < partitionPerNodeCount; pid++)
                {
                    maxTimeCompute = std::max(maxTimeCompute, *recv_buffer_maxTime_compute[nid][pid]);
                    maxTimeMerge = std::max(maxTimeMerge, *recv_buffer_maxTime_merge[nid][pid]);
                    mergeCycle = std::max(mergeCycle, *recv_buffer_mergeCycle[nid][pid]);
                }
            }
            testUtilClient.time_VC_C = maxTimeCompute;
            testUtilClient.time_VC_M = maxTimeMerge;
            testUtilClient.mergeCycle_vertex = mergeCycle;

            //=== VC end ===
            
            //Test
            std::cout << "time_VC_C:" << testUtilClient.time_VC_C << std::endl;
            std::cout << "time_VC_M:" << testUtilClient.time_VC_M << std::endl;
            //Test end

            // get v2v, has to before VM
            testUtilClient.GetV2V(*gd, aggDimension[cuboidCount - 1]);

            //=== VM begin ===
            testUtilClient.VertexMerge(cuboidSet[cuboidCount - 1], aggVListSet, aggDimension.at(cuboidCount - 1));
            //=== VM end ===

            //Test
            std::cout << "time_VM:" << testUtilClient.time_VM << std::endl;
            //Test end

            // EC begin

            // transfrom edge data into send_buffer
            for (int nid = 0; nid < nodeCount; nid++)
            {
                for (int pid = 0; pid < partitionPerNodeCount; pid++)
                {
                    int subg_id = nid * partitionPerNodeCount + pid;
                    //eSet
                    memcpy(send_buffer_graph_element[nid][pid], &subGraphSet[subg_id].eList[0], subGraphSet[subg_id].eCount * sizeof(Edge));
                    //v2v, the first "int" is length
                    *(int*)((long long)send_buffer_graph_other[nid][pid]) = testUtilClient.v2v.size();
                    int i = 0;
                    for (auto it = testUtilClient.v2v.begin(); it != testUtilClient.v2v.end(); it++)
                    {
                        *(int*)((long long)send_buffer_graph_other[nid][pid] + i * sizeof(int) + sizeof(int)) = *it;
                        i++;
                    }
                    //eCount
                    *send_buffer_element_count[nid][pid] = subGraphSet[subg_id].eCount;
                }
            }

            //send data in send_buffer to each worker
            for (int nid = 0; nid < workerCount; nid++)
            {
                for (int pid = 0; pid < partitionPerNodeCount; pid++)
                {
                    //eSet: Tag = 0 + 10 * pid
                    MPI_Send(send_buffer_graph_element[nid][pid], send_buffer_graph_element_size, MPI_BYTE, nid + 1, 0 + 10 * pid, MPI_COMM_WORLD);
                    //v2v: Tag = 1 + 10 * pid
                    MPI_Send(send_buffer_graph_other[nid][pid], send_buffer_graph_other_size, MPI_BYTE, nid + 1, 1 + 10 * pid, MPI_COMM_WORLD);
                    //eCount: Tag = 2 + 10 * pid
                    MPI_Send(send_buffer_element_count[nid][pid], 1, MPI_INT, nid + 1, 2 + 10 * pid, MPI_COMM_WORLD);
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);    //3-after sending edge data

            // receive edge partitions data from all worker 
            for (int nid = 0; nid < workerCount; nid++)
            {
                for (int pid = 0; pid < partitionPerNodeCount; pid++)
                {
                    //agg eSet: Tag = 0 + pid * 10
                    MPI_Recv(recv_buffer_graph_element[nid][pid], recv_buffer_graph_element_size, MPI_BYTE, nid + 1, 0 + pid * 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    //aggCount: Tag = 1 + pid * 10
                    MPI_Recv(recv_buffer_element_count[nid][pid], 1, MPI_INT, nid + 1, 1 + pid * 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    //maxTimeCompute: Tag = 2 + pid * 10
                    MPI_Recv(recv_buffer_maxTime_compute[nid][pid], 1, MPI_INT, nid + 1, 2 + pid * 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    //maxTimeMerge: Tag = 3 + pid * 10
                    MPI_Recv(recv_buffer_maxTime_merge[nid][pid], 1, MPI_INT, nid + 1, 3 + pid * 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    //mergeCycle: Tag = 4 + pid * 10
                    MPI_Recv(recv_buffer_mergeCycle[nid][pid], 1, MPI_INT, nid + 1, 4 + pid * 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }

            // transform the data into aggEListSet
            for (int nid = 0; nid < nodeCount; nid++)
            {
                for (int pid = 0; pid < partitionPerNodeCount; pid++)
                {
                    int aggCount = *recv_buffer_element_count[nid][pid];
                    int subg_id = pid + nid * partitionPerNodeCount;
                    for (int j = 0; j < aggCount; j++)
                    {
                        AggEdge tmp = *(AggEdge*)((long long)recv_buffer_graph_element[nid][pid] + j * sizeof(AggEdge));
                        aggEListSet[subg_id].push_back(tmp);
                    }
                }
            }

            // record the time and mergeCycle
            maxTimeCompute = 0;
            maxTimeMerge = 0;
            mergeCycle = 0;
            for (int nid = 0; nid < nodeCount; nid++)
            {
                for (int pid = 0; pid < partitionPerNodeCount; pid++)
                {
                    maxTimeCompute = std::max(maxTimeCompute, *recv_buffer_maxTime_compute[nid][pid]);
                    maxTimeMerge = std::max(maxTimeMerge, *recv_buffer_maxTime_merge[nid][pid]);
                    mergeCycle = std::max(mergeCycle, *recv_buffer_mergeCycle[nid][pid]);
                }
            }
            testUtilClient.time_EC_C = maxTimeCompute;
            testUtilClient.time_EC_M = maxTimeMerge;
            testUtilClient.mergeCycle_edge = mergeCycle;
            
            //=== EC end ===

            //Test
            std::cout << "time_EC_C:" << testUtilClient.time_EC_C << std::endl;
            std::cout << "time_EC_M:" << testUtilClient.time_EC_M << std::endl;
            //Test end

            
            // EM begin
            testUtilClient.EdgeMerge(cuboidSet[cuboidCount - 1], aggEListSet);
            // EM end

            //Test
            std::cout << "time_EM:" << testUtilClient.time_EM << std::endl;
            //Test end

            testUtilClient.AggEnd();

            //Test
            std::cout << "time:" << testUtilClient.time << std::endl;
            std::cout << "mergeCycle_vertex:" << testUtilClient.mergeCycle_vertex << std::endl;
            std::cout << "mergeCycle_edge:" << testUtilClient.mergeCycle_edge << std::endl;
            //Test end

            //test
            testUtilClient.GetOptimalMergeCycle((*gd), aggDimension[cuboidCount - 1], testUtilClient.currentPartition, testUtilClient.currentThread);
        }

        //release MPI buffer
        for (int nid = 0; nid < nodeCount; nid++)
        {
            for (int pid = 0; pid < partitionPerNodeCount; pid++)
            {
                delete[] send_buffer_graph_element[nid][pid];
                delete[] recv_buffer_graph_element[nid][pid];
                delete[] send_buffer_graph_other[nid][pid];
                delete send_buffer_element_count[nid][pid];
                delete recv_buffer_element_count[nid][pid];
                delete recv_buffer_maxTime_compute[nid][pid];
                delete recv_buffer_maxTime_merge[nid][pid];
                delete recv_buffer_mergeCycle[nid][pid];
            }
        }

        //result check
        std::cout << "AggGraph - ";
        std::cout << "Vertex: " << cuboidSet.at(cuboidID - 1).vCount << ", Edge: " << cuboidSet.at(cuboidID - 1).eCount << std::endl;
    }

    //Worker Process
    else
    {
        //Buffer between master and worker
        std::vector<byte*> recv_buffer_graph_element(partitionPerNodeCount);    //vSet or eSet
        std::vector<byte*> recv_buffer_graph_other(partitionPerNodeCount);      //attrSet measure or v2v
        std::vector<byte*> send_buffer_graph_element(partitionPerNodeCount);    //AggVertex Set or AggEdge Set
        std::vector<int*> recv_buffer_element_count(partitionPerNodeCount);     //vCount or eCount
        std::vector<int*> send_buffer_element_count(partitionPerNodeCount);     //aggCount
        std::vector<int*> send_buffer_maxTime_compute(partitionPerNodeCount);   //maxTimeCompute
        std::vector<int*> send_buffer_maxTime_merge(partitionPerNodeCount);     //maxTimeMerge
        std::vector<int*> send_buffer_mergeCycle(partitionPerNodeCount);        //mergeCycle

        int recv_buffer_graph_element_size = std::max(10 * vCount_perPartition * sizeof(Vertex), 10 * eCount_perPartition * sizeof(Edge));
        int send_buffer_graph_element_size = std::max((unsigned)(10 * sizeof(AggVertex) * vCount_perPartition), (unsigned)(10 * eCount_perPartition * sizeof(AggEdge)));
        int recv_buffer_graph_other_size = std::max((unsigned)(10 * sizeof(CubeAggVertexValue) * vCount_perPartition), (unsigned)(10 * (vCount + 1) * sizeof(int)));

        for (int pid = 0; pid < partitionPerNodeCount; pid++)
        {
            recv_buffer_graph_element[pid] = new byte[recv_buffer_graph_element_size];
            send_buffer_graph_element[pid] = new byte[send_buffer_graph_element_size];
            recv_buffer_graph_other[pid] = new byte[recv_buffer_graph_other_size];
            recv_buffer_element_count[pid] = new int;
            send_buffer_element_count[pid] = new int;
            send_buffer_maxTime_compute[pid] = new int;
            send_buffer_maxTime_merge[pid] = new int;
            send_buffer_mergeCycle[pid] = new int;
        }

        auto testUtilClient = CubeAgg<CubeAggVertexValue, int>(); //Main Worker
        testUtilClient.currentThread = threadCount;
        testUtilClient.currentPartition = partitionPerNodeCount;
        testUtilClient.dCount = dCount;

        auto clientVec = std::vector<UtilClient<CubeAggVertexValue, int>>(); //UtilClient connect UtilServer
        for (int i = 0; i < partitionPerNodeCount; i++)
            clientVec.push_back(UtilClient<CubeAggVertexValue, int>(vCount, eCount, 1000, i));
        //Client Connect Test
        int chk = 0;
        for (int i = 0; i < partitionPerNodeCount && chk != -1; i++)
        {
            chk = clientVec.at(i).connect();
            if (chk == -1)
            {
                std::cout << "Cannot establish the connection with server correctly" << std::endl;
                return 2;
            }
            clientVec.at(i).disconnect();
        }

        //Test
        std::cout << "Init finished" << std::endl;
        //Test end

        //ApplyD
        while (cuboidCount < pow(2, dCount))
        {
            MPI_Barrier(MPI_COMM_WORLD);    //1-beginning

            cuboidCount++;

            if (cuboidCount != cuboidID)	continue;

            testUtilClient.AggInit();

            //receive vertex data from master
            for (int pid = 0; pid < partitionPerNodeCount; pid++)
            {
                //vSet: Tag = 0 + 10 * pid
                MPI_Recv(recv_buffer_graph_element[pid], recv_buffer_graph_element_size, MPI_BYTE, 0, 0 + pid * 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                //vertexImf: Tag = 1 + 10 * pid
                MPI_Recv(recv_buffer_graph_other[pid], recv_buffer_graph_other_size, MPI_BYTE, 0, 1 + pid * 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                //vCount: Tag = 2 + 10 * pid
                MPI_Recv(recv_buffer_element_count[pid], 1, MPI_INT, 0, 2 + pid * 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            std::vector<CubeAggGraph> subGraphSet(partitionPerNodeCount);

            //transfrom data into subGraphSet
            for (int pid = 0; pid < partitionPerNodeCount; pid++)
            {
                // vCount
                subGraphSet[pid].vCount = *recv_buffer_element_count[pid];

                // vSet
                for (int i = 0; i < subGraphSet[pid].vCount; i++)
                {
                    Vertex tmp_vertex = *(Vertex*)((long long)recv_buffer_graph_element[pid] + i * sizeof(Vertex));
                    subGraphSet[pid].vList.push_back(tmp_vertex);
                }

                // vertexImf
                int CubeAggVertexValueSize = sizeof(CubeAggVertexValue);
                for (int i = 0; i < subGraphSet[pid].vCount; i++)
                {
                    long long address = ((long long)recv_buffer_graph_other[pid] + i * CubeAggVertexValueSize);
                    CubeAggVertexValue tmp = *(CubeAggVertexValue*)address;
                    subGraphSet[pid].verticesValue.push_back(tmp);
                }
            }

            for (int i = 0; i < partitionPerNodeCount; i++)
            {
                clientVec[i].vCount = subGraphSet[i].vCount;
            }

            std::vector<std::vector<AggVertex>> aggVListSet(partitionPerNodeCount);  // intermediate agg sub graph set
            std::vector<std::vector<AggEdge>> aggEListSet(partitionPerNodeCount);  // intermediate agg sub graph set

            // VC begin
            auto futList = new std::future<void>[partitionPerNodeCount];

            //type sub_vCount sub_eCount aggDimension
            int** initVSet = new int* [partitionPerNodeCount];
            for (int i = 0; i < partitionPerNodeCount; i++)
            {
                initVSet[i] = new int[3 + dCount];
                initVSet[i][0] = 0;    // type = VC
                initVSet[i][1] = subGraphSet[i].vCount;
                initVSet[i][2] = 1;    // sub_eCount = 1 temperarily
                for (int j = 0; j < dCount; j++)    initVSet[i][j + 3] = aggDimension[cuboidCount - 1][j];
            }

            // create temp space for vValue
            byte** vValue = new byte * [partitionPerNodeCount];
            int CubeAgg_Value_size = sizeof(CubeAggVertexValue);
            for (int i = 0; i < partitionPerNodeCount; i++)
            {
                // V1 V2 ...
                int v_count = subGraphSet[i].vCount;
                //test
                vValue[i] = new byte[CubeAgg_Value_size * vCount * 10];   //measure and attr

                for (int j = 0; j < v_count; j++)
                {
                    long long address = (long long)vValue[i] + CubeAgg_Value_size * j;
                    *(CubeAggVertexValue*)address = subGraphSet[i].verticesValue[j];
                }
            }

            Edge* tmp_edge = new Edge[eCount];  //temp edge
            Vertex* tmp_vertex = new Vertex[vCount];    //temp vertex
            bool* tmp_filteredV = new bool[vCount];
            int tmp_filteredVCount = 0;

            //assign works to servers
            for (int i = 0; i < partitionPerNodeCount; i++)
            {
                Vertex* vSet = subGraphSet[i].vCount > 0 ? &subGraphSet[i].vList[0] : tmp_vertex;   // some time vCount = 0
                Edge* eSet = tmp_edge;
                bool* filteredV = tmp_filteredV;
                std::future<void> tmpFut = std::async(testFut<CubeAggVertexValue, int>, &clientVec.at(i),
                    subGraphSet[i].vCount, subGraphSet[i].eCount, (CubeAggVertexValue*)vValue[i], vSet, eSet,
                    initVSet[i], filteredV, tmp_filteredVCount);
                futList[i] = std::move(tmpFut);
            }
            for (int i = 0; i < partitionPerNodeCount; i++)
                futList[i].get();

            //Retrieve data
            std::vector<long long> maxTimeCompute(partitionPerNodeCount);
            std::vector<long long> maxTimeMerge(partitionPerNodeCount);
            std::vector<long long> mergeCycle(partitionPerNodeCount);

            for (int i = 0; i < partitionPerNodeCount; i++)
            {
                clientVec.at(i).connect();

                //Collect data: maxTime_1 maxTime_2 mergeCycle aggCount aggV1 aggV2 ...
                maxTimeCompute[i] = *(long long*)clientVec[i].mValues;
                maxTimeMerge[i] = *(long long*)((long long)clientVec[i].mValues + sizeof(long long));
                mergeCycle[i] = *(int*)((long long)clientVec[i].mValues + 2 * sizeof(long long));
                int aggCount = *(int*)((long long)clientVec[i].mValues + 2 * sizeof(long long) + sizeof(int));
                int offset = 2 * sizeof(long long) + 2 * sizeof(int);

                int aggV_MT_size = sizeof(AggVertex);
                for (int j = 0; j < aggCount; j++)
                {
                    long long address = (long long)clientVec[i].mValues + j * aggV_MT_size + offset;
                    AggVertex tmp = *(AggVertex*)address;
                    aggVListSet[i].push_back(tmp);
                }
                clientVec.at(i).disconnect();
            }

            //transform data into send_buffer
            for (int pid = 0; pid < partitionPerNodeCount; pid++)
            {
                //agg vSet
                int AggVertexSize = sizeof(AggVertex);
                memcpy(send_buffer_graph_element[pid], &aggVListSet[pid][0], aggVListSet[pid].size()* AggVertexSize);                                          
                //aggCount
                *send_buffer_element_count[pid] = aggVListSet[pid].size();
                //maxTimeCompute
                *send_buffer_maxTime_compute[pid] = maxTimeCompute[pid];
                //maxTimeMerge
                *send_buffer_maxTime_merge[pid] = maxTimeMerge[pid];
                //mergeCycle
                *send_buffer_mergeCycle[pid] = mergeCycle[pid];
            }

            MPI_Barrier(MPI_COMM_WORLD);    //2-before sending agg_vertex data

            //send agg_vertex partitions data to master
            for (int pid = 0; pid < partitionPerNodeCount; pid++)
            {
                //agg vSet: Tag = 0 + pid * 10
                MPI_Send(send_buffer_graph_element[pid], send_buffer_graph_element_size, MPI_BYTE, 0, 0 + pid * 10, MPI_COMM_WORLD);
                //aggCount: Tag = 1 + pid * 10
                MPI_Send(send_buffer_element_count[pid], 1, MPI_INT, 0, 1 + pid * 10, MPI_COMM_WORLD);
                //maxTimeCompute: Tag = 2 + pid * 10
                MPI_Send(send_buffer_maxTime_compute[pid], 1, MPI_INT, 0, 2 + pid * 10, MPI_COMM_WORLD);
                //maxTimeMerge: Tag = 3 + pid * 10
                MPI_Send(send_buffer_maxTime_merge[pid], 1, MPI_INT, 0, 3 + pid * 10, MPI_COMM_WORLD);
                //mergeCycle: Tag = 4 + pid * 10
                MPI_Send(send_buffer_mergeCycle[pid], 1, MPI_INT, 0, 4 + pid * 10, MPI_COMM_WORLD);
            }

            //=== VC end ===

            //=== EC begin ===

            //receive graph partitions data from master
            for (int pid = 0; pid < partitionPerNodeCount; pid++)
            {
                //eSet: Tag = 0 + 10 * pid
                MPI_Recv(recv_buffer_graph_element[pid], recv_buffer_graph_element_size, MPI_BYTE, 0, 0 + pid * 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                //v2v: Tag = 1 + 10 * pid , the first "int" is the lenth
                MPI_Recv(recv_buffer_graph_other[pid], recv_buffer_graph_other_size, MPI_BYTE, 0, 1 + pid * 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                //vCount: Tag = 2 + 10 * pid
                MPI_Recv(recv_buffer_element_count[pid], 1, MPI_INT, 0, 2 + pid * 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            //transfrom data into subGraphSet
            for (int pid = 0; pid < partitionPerNodeCount; pid++)
            {
                subGraphSet[pid].eCount = *recv_buffer_element_count[pid];
                for (int i = 0; i < subGraphSet[pid].eCount; i++)
                {
                    Edge tmp_edge = *(Edge*)((long long)recv_buffer_graph_element[pid] + i * sizeof(Edge));
                    subGraphSet[pid].eList.push_back(tmp_edge);
                }
            }

            for (int i = 0; i < partitionPerNodeCount; i++)
            {
                clientVec[i].eCount = subGraphSet[i].eCount;
            }

            //type sub_vCount sub_eCount aggDimension
            for (int i = 0; i < partitionPerNodeCount; i++)
            {
                initVSet[i][0] = 1;// type - EC
                initVSet[i][2] = subGraphSet[i].eCount;
            }

            // put v2v into vValue
            for (int i = 0; i < partitionPerNodeCount; i++)
            {
                int v2v_size = *(int*)((long long)recv_buffer_graph_other[i]);
                memcpy(vValue[i], (void*)((long long)recv_buffer_graph_other[i]+sizeof(int)), v2v_size * sizeof(int));
            }

            //assign works to servers
            for (int i = 0; i < partitionPerNodeCount; i++)
            {
                Vertex* vSet = tmp_vertex;
                Edge* eSet = subGraphSet[i].eCount > 0 ? &subGraphSet[i].eList[0] : tmp_edge;
                bool* filteredV = tmp_filteredV;
                std::future<void> tmpFut = std::async(testFut<CubeAggVertexValue, int>, &clientVec.at(i),
                    subGraphSet[i].vCount, subGraphSet[i].eCount, (CubeAggVertexValue*)vValue[i], vSet, eSet,
                    initVSet[i], filteredV, tmp_filteredVCount);
                futList[i] = std::move(tmpFut);
            }
            for (int i = 0; i < partitionPerNodeCount; i++)
                futList[i].get();


            //Retrieve data
            for (int i = 0; i < partitionPerNodeCount; i++)
            {
                clientVec.at(i).connect();

                //Collect data: maxTime1 maxTime2 mergeCycle aggCount aggE1 aggE2 ...
                maxTimeCompute[i] = *(long long*)clientVec[i].mValues;
                maxTimeMerge[i] = *(long long*)((long long)clientVec[i].mValues + sizeof(long long));
                mergeCycle[i] = *(int*)((long long)clientVec[i].mValues + 2 * sizeof(long long));
                int aggCount = *(int*)((long long)clientVec[i].mValues + 2 * sizeof(long long) + sizeof(int));
                long long address = (long long)clientVec[i].mValues + 2 * sizeof(long long) + 2 * sizeof(int);
                for (int j = 0; j < aggCount; j++)
                {
                    auto tmp = *(AggEdge*)(address + j * sizeof(AggEdge));
                    aggEListSet[i].push_back(tmp);
                }

                clientVec.at(i).disconnect();
            }
            // EC end

            //transform data into send_buffer
            for (int pid = 0; pid < partitionPerNodeCount; pid++)
            {
                //agg eSet
                memcpy(send_buffer_graph_element[pid], &aggEListSet[pid][0], aggEListSet[pid].size() * sizeof(AggEdge));
                //aggCount
                *send_buffer_element_count[pid] = aggEListSet[pid].size();
                //maxTimeCompute
                *send_buffer_maxTime_compute[pid] = maxTimeCompute[pid];
                //maxTimeMerge
                *send_buffer_maxTime_merge[pid] = maxTimeMerge[pid];
                //mergeCycle
                *send_buffer_mergeCycle[pid] = mergeCycle[pid];
            }

            MPI_Barrier(MPI_COMM_WORLD);    //3-before sending agg_edge data

            //send agg_edge partitions data to master
            for (int pid = 0; pid < partitionPerNodeCount; pid++)
            {
                //agg eSet: Tag = 0 + pid * 10
                MPI_Send(send_buffer_graph_element[pid], send_buffer_graph_element_size, MPI_BYTE, 0, 0 + pid * 10, MPI_COMM_WORLD);
                //aggCount: Tag = 1 + pid * 10
                MPI_Send(send_buffer_element_count[pid], 1, MPI_INT, 0, 1 + pid * 10, MPI_COMM_WORLD);
                //maxTimeCompute: Tag = 2 + pid * 10
                MPI_Send(send_buffer_maxTime_compute[pid], 1, MPI_INT, 0, 2 + pid * 10, MPI_COMM_WORLD);
                //maxTimeMerge: Tag = 3 + pid * 10
                MPI_Send(send_buffer_maxTime_merge[pid], 1, MPI_INT, 0, 3 + pid * 10, MPI_COMM_WORLD);
                //mergeCycle: Tag = 4 + pid * 10
                MPI_Send(send_buffer_mergeCycle[pid], 1, MPI_INT, 0, 4 + pid * 10, MPI_COMM_WORLD);
            }

            // release memory
            for (int i = 0; i < partitionPerNodeCount; i++)
            {
                delete[] initVSet[i];
                delete[] vValue[i];
            }
            delete[] vValue;
            delete[] initVSet;
            delete[] tmp_filteredV;
            delete[] tmp_edge;
            delete[] tmp_vertex;
        }

        for (int i = 0; i < partitionPerNodeCount; i++) clientVec.at(i).shutdown();

        //release MPI buffer
        for (int pid = 0; pid < partitionPerNodeCount; pid++)
        {
            delete[] recv_buffer_graph_element[pid];
            delete[] send_buffer_graph_element[pid];
            delete[] recv_buffer_graph_other[pid];
            delete recv_buffer_element_count[pid];
            delete send_buffer_element_count[pid];
            delete send_buffer_maxTime_compute[pid];
            delete send_buffer_maxTime_merge[pid];
            delete send_buffer_mergeCycle[pid];
        }
    }
    
    //test
    printf("process:%d has finished\n", world_rank);

    MPI_Finalize();
}