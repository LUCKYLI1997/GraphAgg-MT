//
// Created by Thoh Testarossa on 2019-03-11.
//

#include "../core/GraphUtil.h"
#include "../algo/BellmanFord/BellmanFord.h"

#include <fstream>
#include <iostream>

int main()
{
    //Read the Graph
    std::ifstream Gin("testGraph.txt");
    if(!Gin.is_open()) {std::cout << "Error! File testGraph.txt not found!" << std::endl; return 1; }

    int vCount, eCount;
    Gin >> vCount >> eCount;

    Graph<double> test = Graph<double>(vCount);
    for(int i = 0; i < eCount; i++)
    {
        int src, dst;
        double weight;

        Gin >> src >> dst >> weight;
        test.insertEdge(src, dst, weight);
    }

    Gin.close();

    auto test2 = test;

    std::vector<int> initVList = std::vector<int>();
    initVList.push_back(1);
    initVList.push_back(2);
    initVList.push_back(4);

    //reflectG() test part
    std::cout << "reflect() test:" << std::endl;

    //BF on original graph
    std::cout << "Result for Original graph:" << std::endl;
    auto executor = BellmanFord<double, double>();
    executor.ApplyD(test, initVList, 4);
    for(int i = 0; i < test.vCount * initVList.size(); i++)
    {
        if(i % initVList.size() == 0) std::cout << i / initVList.size() << ": ";
        std::cout << "(" << initVList.at(i % initVList.size()) << " -> " << test.verticesValue.at(i) << ")";
        if(i % initVList.size() == initVList.size() - 1) std::cout << std::endl;
    }
    std::cout << "******************************************************************" << std::endl;

    //BF on reflected graph
    std::cout << "Result for reflected graph:" << std::endl;
    auto executor2 = BellmanFord<double, double>();
    auto reflectIndex = std::vector<int>();
    auto reversedIndex = std::vector<int>();
    auto r_test2 = executor2.reflectG(test2, test2.eList, reflectIndex, reversedIndex);
    std::cout << "Reflection generated:" << std::endl;
    for(int i = 0; i < reflectIndex.size(); i++)
    {
        if(reflectIndex.at(i) != NO_REFLECTION)
            std::cout << "(" << i << ", " << reflectIndex.at(i) << ")" << " ";
    }
    std::cout << std::endl;
    auto r_initVList = std::vector<int>();
    for(int i = 0; i < initVList.size(); i++) r_initVList.emplace_back(reversedIndex.at(initVList.at(i)));
    executor2.ApplyD(r_test2, r_initVList, 4);
    for(int i = 0; i < r_test2.vCount * r_initVList.size(); i++)
    {
        if(i % r_initVList.size() == 0) std::cout << i / r_initVList.size() << ": ";
        std::cout << "(" << r_initVList.at(i % r_initVList.size()) << " -> " << r_test2.verticesValue.at(i) << ")";
        if(i % r_initVList.size() == r_initVList.size() - 1) std::cout << std::endl;
    }
    std::cout << "******************************************************************" << std::endl;

    //BF result on deflected graph
    std::cout << "Result for deflected graph:" << std::endl;
    auto executor3 = BellmanFord<double, double>();
    auto AVSet = std::set<int>();
    executor3.Init(test2.vCount, test2.eCount, initVList.size());
    executor3.GraphInit(test2, AVSet, initVList);
    executor3.Deploy(test2.vCount, test2.eCount, initVList.size());
    for(int i = 0; i < r_test2.vCount * initVList.size(); i++)
        test2.verticesValue.at(reflectIndex[i / initVList.size()] * initVList.size() + i % initVList.size()) = r_test2.verticesValue.at(i);
    for(int i = 0; i < test2.vCount * initVList.size(); i++)
    {
        if(i % initVList.size() == 0) std::cout << i / initVList.size() << ": ";
        std::cout << "(" << initVList.at(i % initVList.size()) << " -> " << test2.verticesValue.at(i) << ")";
        if(i % initVList.size() == initVList.size() - 1) std::cout << std::endl;
    }

    return 0;
}