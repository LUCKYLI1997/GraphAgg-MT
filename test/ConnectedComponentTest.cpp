//
// Created by Thoh Testarossa on 2019-08-12.
//

#include "../algo/ConnectedComponent/ConnectedComponent.h"

#include <iostream>
#include <fstream>

int main()
{
    //Read the Graph
    std::ifstream Gin("testGraph.txt");
    if(!Gin.is_open()) {std::cout << "Error! File testGraph.txt not found!" << std::endl; return 1; }

    int vCount, eCount;
    Gin >> vCount >> eCount;

    Graph<int> test = Graph<int>(vCount);
    for(int i = 0; i < eCount; i++)
    {
        int src, dst;
        double weight;

        Gin >> src >> dst >> weight;
        test.insertEdge(src, dst, weight);
    }

    Gin.close();

    std::vector<int> initVList = std::vector<int>();

    ConnectedComponent<int, int> executor = ConnectedComponent<int, int>();
    //executor.Apply(test, initVList);
    executor.ApplyD(test, initVList, 4);

    for(int i = 0; i < test.vCount; i++)
        std::cout << i << ": " << test.verticesValue.at(i) << std::endl;
}
