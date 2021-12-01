//
// Created by cave-g-f on 2019-05-24.
//

#include "../algo/LabelPropagation/LabelPropagation.h"

#include <iostream>
#include <fstream>

int main()
{
    //Read the Graph
    std::ifstream Gin("testGraph.txt");
    if(!Gin.is_open()) {std::cout << "Error! File testGraph.txt not found!" << std::endl; return 1; }

    int vCount, eCount;
    Gin >> vCount >> eCount;

    std::vector<int> initVList = std::vector<int>();

    Graph<std::pair<int, int>> test = Graph<std::pair<int, int>>(vCount);
    for(int i = 0; i < eCount; i++)
    {
        int src, dst;
        double weight;

        Gin >> src >> dst >> weight;
        test.insertEdge(src, dst, weight);
    }

    Gin.close();

    auto executor = LabelPropagation<std::pair<int, int>, std::pair<int, int>>();
    executor.ApplyD(test, initVList, 4);
}

