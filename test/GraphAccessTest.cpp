//
// Created by Thoh Testarossa on 2019-03-08.
//

#include "../core/Graph.h"

#include <iostream>
#include <fstream>

int main()
{
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

    std::ofstream Gout("testOutput.txt");
    Gout << test.vCount << " " << test.eCount << std::endl;
    for(int i = 0; i < eCount; i++) Gout << test.eList[i].src << " "
                                         << test.eList[i].dst << " "
                                         << test.eList[i].weight << std::endl;

    Gout.close();

    return 0;
}