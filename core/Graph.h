//
// Created by Thoh Testarossa on 2019-03-08.
//

#pragma once

#ifndef GRAPH_ALGO_GRAPH_H
#define GRAPH_ALGO_GRAPH_H

#include "../include/deps.h"
#include "../core/AbstractGraph.h"

template <typename VertexValueType>
class Graph : public AbstractGraph
{
public:
    Graph(int vCount);
    Graph(const std::vector<Vertex> &vSet, const std::vector<Edge> &eSet, const std::vector<VertexValueType> &verticesValue);
    Graph(int vCount, int eCount, int *eSrcSet, int *eDstSet, double *eWeightSet, bool *AVCheckSet);
    Graph(int vCount, int eCount, int numOfInitV, const int *initVSet, int *eSrcSet, int *eDstSet, double *eWeightSet, bool *AVCheckSet);

    std::vector<VertexValueType> verticesValue;
};

#endif //GRAPH_ALGO_GRAPH_H
