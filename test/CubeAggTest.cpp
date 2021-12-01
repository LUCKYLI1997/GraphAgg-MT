#include "../algo/CubeAgg/CubeAgg.h"

#include <iostream>
#include <fstream>
#include <math.h>

int main(int argc, char* argv[])
{
    if (argc != 5)
    {
        std::cout << "Usage:" << std::endl << "./algo_CubeAggTest dCount partitionCount threadCount CuboidID" << std::endl;
        return 1;
    }

    int dCount = atoi(argv[1]);
    int partitionCount = atoi(argv[2]);
    int threadCount = atoi(argv[3]);
    int cuboidID = atoi(argv[4]);

    //Parameter check
    if (dCount <= 0 || partitionCount <= 0 || threadCount <= 0)
    {
        std::cout << "Parameter illegal" << std::endl;
        return 3;
    }

    // Read the Graph
    std::ifstream GinVertex("testGraphVertex.txt");
    if (!GinVertex.is_open()) { std::cout << "Error! File testGraphVertex.txt not found!" << std::endl; return 1; }
    std::ifstream GinEdge("testGraphEdge.txt");
    if (!GinEdge.is_open()) { std::cout << "Error! File testGraphEdge.txt not found!" << std::endl; return 1; }

    int vCount, eCount;
    GinVertex >> vCount;
    GinEdge >> eCount;

    CubeAggGraph test;
    test.vCount = vCount;
    test.vList = std::vector<Vertex>(vCount);
    test.verticesValue = std::vector<CubeAggVertexValue>(vCount);
    for (int i = 0; i < vCount; i++)
    {
        int vertexID;
        int measure;
        std::vector<std::string> d(dCount);
        GinVertex >> vertexID;
        for (int j = 0; j < dCount; j++)
        {
            GinVertex >> test.verticesValue.at(vertexID).dimension_ptr[j];
        }
        GinVertex >> test.verticesValue.at(vertexID).measure;
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
    //std::vector<int> dgList_4 = { -1,0,0,0,0,2,0,2,0,2, 0, 6, 9, 6, 9, 6 };
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

    std::vector<CubeAggGraph> agg_gSet(pow(2, dCount));

    CubeAgg<CubeAggVertexValue, int> executor = CubeAgg<CubeAggVertexValue, int>();
    executor.currentThread = threadCount;
    executor.currentPartition = partitionCount;
    executor.dCount = dCount;

    executor.ApplyD(test, aggDimension, agg_gSet, dgList, cuboidID);

    //result check
    std::cout << "AggGraph - ";
    std::cout << "Vertex: " << agg_gSet.at(cuboidID-1).vCount << ", Edge: " << agg_gSet.at(cuboidID - 1).eCount << std::endl;

    return 1;
}