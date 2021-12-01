//
// Created by Thoh Testarossa on 2019-08-17.
//

#pragma once

#ifndef GRAPH_ALGO_STRONGLYCONNECTEDCOMPONENT_H
#define GRAPH_ALGO_STRONGLYCONNECTEDCOMPONENT_H

#include "../../core/GraphUtil.h"

//Stage 1: Distributed DFS used to calculate "finish order"
template <typename VertexValueType, typename MessageValueType>
class StronglyConnectedComponent_stage_1 : public GraphUtil<VertexValueType, MessageValueType>
{
public:
    StronglyConnectedComponent_stage_1();

    int MSGApply(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertice, const MessageSet<MessageValueType> &mSet) override;
    int MSGGenMerge(const Graph<VertexValueType> &g, const std::vector<int> &initVSet, const std::set<int> &activeVertice, MessageSet<MessageValueType> &mSet) override;

    //Unified interface but actually algo_BellmanFord didn't use this form
    int MSGApply_array(int vCount, int eCount, Vertex *vSet, int numOfInitV, const int *initVSet, VertexValueType *vValues, MessageValueType *mValues) override;
    int MSGGenMerge_array(int vCount, int eCount, const Vertex *vSet, const Edge *eSet, int numOfInitV, const int *initVSet, const VertexValueType *vValues, MessageValueType *mValues) override;

    void MergeGraph(Graph<VertexValueType> &g, const std::vector<Graph<VertexValueType>> &subGSet,
                    std::set<int> &activeVertices, const std::vector<std::set<int>> &activeVerticeSet,
                    const std::vector<int> &initVList) override;

    void Init(int vCount, int eCount, int numOfInitV) override;
    void GraphInit(Graph<VertexValueType> &g, std::set<int> &activeVertices, const std::vector<int> &initVList) override;
    void Deploy(int vCount, int eCount, int numOfInitV) override;
    void Free() override;

    void ApplyStep(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertices);
    void Apply(Graph<VertexValueType> &g, const std::vector<int> &initVList);

    void ApplyD(Graph<VertexValueType> &g, const std::vector<int> &initVList, int partitionCount);

protected:
    int numOfInitV;
};

//Stage 2: BFS-based coloring used to calculate connectivity (named SCC)
//The same as algo_CC
template <typename VertexValueType, typename MessageValueType>
class StronglyConnectedComponent_stage_2 : public GraphUtil<VertexValueType, MessageValueType>
{
    StronglyConnectedComponent_stage_2();

    int MSGApply(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertice, const MessageSet<MessageValueType> &mSet) override;
    int MSGGenMerge(const Graph<VertexValueType> &g, const std::vector<int> &initVSet, const std::set<int> &activeVertice, MessageSet<MessageValueType> &mSet) override;

    //Unified interface but actually algo_BellmanFord didn't use this form
    int MSGApply_array(int vCount, int eCount, Vertex *vSet, int numOfInitV, const int *initVSet, VertexValueType *vValues, MessageValueType *mValues) override;
    int MSGGenMerge_array(int vCount, int eCount, const Vertex *vSet, const Edge *eSet, int numOfInitV, const int *initVSet, const VertexValueType *vValues, MessageValueType *mValues) override;

    void MergeGraph(Graph<VertexValueType> &g, const std::vector<Graph<VertexValueType>> &subGSet,
                    std::set<int> &activeVertices, const std::vector<std::set<int>> &activeVerticeSet,
                    const std::vector<int> &initVList) override;

    void Init(int vCount, int eCount, int numOfInitV) override;
    void GraphInit(Graph<VertexValueType> &g, std::set<int> &activeVertices, const std::vector<int> &initVList) override;
    void Deploy(int vCount, int eCount, int numOfInitV) override;
    void Free() override;

    void ApplyStep(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertices);
    void Apply(Graph<VertexValueType> &g, const std::vector<int> &initVList);

    void ApplyD(Graph<VertexValueType> &g, const std::vector<int> &initVList, int partitionCount);

protected:
    int numOfInitV;
};

#endif //GRAPH_ALGO_STRONGLYCONNECTEDCOMPONENT_H
