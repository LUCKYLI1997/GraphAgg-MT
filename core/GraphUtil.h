//
// Created by Thoh Testarossa on 2019-03-08.
//

#pragma once

#ifndef GRAPH_ALGO_GRAPHUTIL_H
#define GRAPH_ALGO_GRAPHUTIL_H

#include "../core/Graph.h"
#include "../core/MessageSet.h"

#define NO_REFLECTION -1

template <typename VertexValueType, typename MessageValueType>
class GraphUtil
{
public:
    virtual int MSGApply(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertices, const MessageSet<MessageValueType> &mSet) = 0;
    virtual int MSGGenMerge(const Graph<VertexValueType> &g, const std::vector<int> &initVSet, const std::set<int> &activeVertices, MessageSet<MessageValueType> &mSet) = 0;

    //For transportation between jni part and processing part by using share memory
    //Also for less data transformation in order to achieve higher performance
    //Data struct Graph is not necessary!?
    virtual int MSGApply_array(int vCount, int eCount, Vertex *vSet, int numOfInitV, const int *initVSet, VertexValueType *vValues, MessageValueType *mValues) = 0;
    virtual int MSGGenMerge_array(int vCount, int eCount, const Vertex *vSet, const Edge *eSet, int numOfInitV, const int *initVSet, const VertexValueType *vValues, MessageValueType *mValues) = 0;

    //Master function
    virtual void Init(int vCount, int eCount, int numOfInitV) = 0;
    virtual void GraphInit(Graph<VertexValueType> &g, std::set<int> &activeVertices, const std::vector<int> &initVList) = 0;
    virtual void Free() = 0;
    virtual void Deploy(int vCount, int eCount, int numOfInitV) = 0;
    virtual void MergeGraph(Graph<VertexValueType> &g, const std::vector<Graph<VertexValueType>> &subGSet,
                            std::set<int> &activeVertices, const std::vector<std::set<int>> &activeVerticeSet,
                            const std::vector<int> &initVList) = 0;
    //virtual void MergeMergedMSG(MessageSet<VertexValueType> &mergedMSG, const std::vector<MessageSet<VertexValueType>> &mergedMSGSet) = 0;

    std::vector<Graph<VertexValueType>> DivideGraphByEdge(const Graph<VertexValueType> &g, int partitionCount);

    //Subgraph reflection-based compression
    int reflect(const std::vector<int> &originalIntList, int originalIntRange, std::vector<int> &reflectIndex, std::vector<int> &reversedIndex);

    Graph<VertexValueType> reflectG(const Graph<VertexValueType> &o_g, const std::vector<Edge> &eSet, std::vector<int> &reflectIndex, std::vector<int> &reversedIndex);
    MessageSet<MessageValueType> reflectM(const MessageSet<MessageValueType> &o_mSet, int vCount, std::vector<int> &reflectIndex, std::vector<int> &reversedIndex);

    //This two parameter is used to initialize memory blocks the program will use
    int totalVValuesCount;
    int totalMValuesCount;
};

#endif //GRAPH_ALGO_GRAPHUTIL_H
