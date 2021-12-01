//
// Created by Thoh Testarossa on 2019-04-06.
//

#include "../core/Graph.h"
#include "../core/GraphUtil.h"
#include "../srv/UtilClient.h"
#include "../srv/UNIX_shm.h"
#include "../srv/UNIX_msg.h"

#include <iostream>
#include <fstream>
#include <vector>

#include <future>
#include <cstring>

template <typename VertexValueType, typename MessageValueType>
void testFut(UtilClient<VertexValueType, MessageValueType> *uc, double *vValues, Vertex *vSet)
{
    uc->connect();
    uc->update(vValues, vSet);
    uc->request();
    uc->disconnect();
}

int main(int argc, char *argv[])
{
    if(argc != 4 && argc != 5)
    {
        std::cout << "Usage:" << std::endl << "./UtilClientTest_BellmanFord vCount eCount numOfInitV [nodeCount]" << std::endl;
        return 1;
    }

    int vCount = atoi(argv[1]);
    int eCount = atoi(argv[2]);
    int numOfInitV = atoi(argv[3]);
    int nodeCount = (argc == 4) ? 1 : atoi(argv[4]);

    //Parameter check
    if(vCount <= 0 || eCount <= 0 || numOfInitV <= 0 || nodeCount <= 0)
    {
        std::cout << "Parameter illegal" << std::endl;
        return 3;
    }

    //Init the Graph
    int *initVSet = new int [numOfInitV];
    double *vValues = new double [vCount * numOfInitV];
    bool *filteredV = new bool [vCount];

    std::vector<Vertex> vSet = std::vector<Vertex>();
    std::vector<Edge> eSet = std::vector<Edge>();

    std::ifstream Gin("testGraph.txt");
    if(!Gin.is_open())
    {
        std::cout << "Error! File testGraph.txt not found!" << std::endl;
        return 4;
    }

    int tmp;
    Gin >> tmp;
    if(vCount != tmp)
    {
        std::cout << "Graph file doesn't match up UtilClient's parameter" << std::endl;
        return 5;
    }
    Gin >> tmp;
    if(eCount != tmp)
    {
        std::cout << "Graph file doesn't match up UtilClient's parameter" << std::endl;
        return 5;
    }

    for(int i = 0; i < vCount * numOfInitV; i++) vValues[i] = INT32_MAX  >> 1;

    //Easy init
    initVSet[0] = 1;
    initVSet[1] = 2;
    initVSet[2] = 3;
    initVSet[3] = 5;
    initVSet[4] = 8;
    for(int i = 0; i < numOfInitV; i++) vValues[initVSet[i] * numOfInitV + i] = 0;

    for(int i = 0; i < vCount; i++) filteredV[i] = false;

    for(int i = 0; i < vCount; i++) vSet.emplace_back(i, false, -1);
    for(int i = 0; i < numOfInitV; i++)
    {
        vSet.at(initVSet[i]).initVIndex = i;
        vSet.at(initVSet[i]).isActive = true;
    }

    for(int i = 0; i < eCount; i++)
    {
        int src, dst;
        double weight;
        Gin >> src >> dst >> weight;
        eSet.emplace_back(src, dst, weight);
    }

    Gin.close();

    //Client Init Data Transfer
    auto clientVec = std::vector<UtilClient<double, double>>();
    for(int i = 0; i < nodeCount; i++)
        clientVec.push_back(UtilClient<double, double>(vCount, ((i + 1) * eCount) / nodeCount - (i * eCount) / nodeCount, numOfInitV, i));
    int chk = 0;
    for(int i = 0; i < nodeCount && chk != -1; i++)
    {
        chk = clientVec.at(i).connect();
        if (chk == -1)
        {
            std::cout << "Cannot establish the connection with server correctly" << std::endl;
            return 2;
        }

        chk = clientVec.at(i).transfer(vValues, &vSet[0], &eSet[(i * eCount) / nodeCount], initVSet, filteredV, vCount);

        if(chk == -1)
        {
            std::cout << "Parameter illegal" << std::endl;
            return 3;
        }

        clientVec.at(i).disconnect();
    }

    bool isActive = false;
    for(int i = 0; i < vCount; i++) isActive |= vSet[i].isActive;

    bool *ret_AVCheckSet = new bool [vCount];
    int iterCount = 0;

    //Test
    std::cout << "Init finished" << std::endl;
    //Test end

    while(isActive)
    {
        //Test
        std::cout << "Processing at iter " << ++iterCount << std::endl;
        //Test end

        for(int i = 0; i < vCount; i++) ret_AVCheckSet[i] = false;

        auto futList = new std::future<void> [nodeCount];
        for(int i = 0; i < nodeCount; i++)
        {
            std::future<void> tmpFut = std::async(testFut<double, double>, &clientVec.at(i), vValues, &vSet[0]);
            futList[i] = std::move(tmpFut);
        }

        for(int i = 0; i < nodeCount; i++)
            futList[i].get();

        //Retrieve data
        for(int i = 0; i < nodeCount; i++)
        {
            clientVec.at(i).connect();

            //Collect data
            for(int j = 0; j < vCount * numOfInitV; j++)
            {
                if (clientVec.at(i).vValues[j] < vValues[j])
                    vValues[j] = clientVec.at(i).vValues[j];
            }

            for(int j = 0; j < vCount; j++)
                ret_AVCheckSet[j] |= clientVec.at(i).vSet[j].isActive;

            clientVec.at(i).disconnect();
        }

        for(int i = 0; i < vCount; i++) vSet[i].isActive = ret_AVCheckSet[i];

        isActive = false;
        for(int i = 0; i < vCount; i++) isActive |= vSet[i].isActive;
    }

    for(int i = 0; i < nodeCount; i++) clientVec.at(i).shutdown();

    //result check
    for(int i = 0; i < vCount * numOfInitV; i++)
    {
        if(i % numOfInitV == 0) std::cout << i / numOfInitV << ": ";
        std::cout << "(" << initVSet[i % numOfInitV] << " -> " << vValues[i] << ")";
        if(i % numOfInitV == numOfInitV - 1) std::cout << std::endl;
    }
}