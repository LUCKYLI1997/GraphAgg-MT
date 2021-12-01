#pragma once
#ifndef CUBEAGG_H
#define CUBEAGG_H

#include <iostream>
#include <unordered_map>
#include <string>
#include <atomic>
#include <math.h>
#include <mutex>
#include <ctime>
#include <algorithm>
#include <thread>
#include <random>
#include <chrono>
#include <string.h>

#include <sys/time.h>
#include "../../core/GraphUtil.h"

#define MAX_NODE_NUMBER 1000000000

class AggVertex
{
public:
    AggVertex();
    AggVertex(char dimension[100], int measure = 0, int TA_count = 0);
    AggVertex(const AggVertex& copy);
    AggVertex& operator=(const AggVertex& object)
    {
        this->measure = (int)object.measure;
        this->TA_count = (int)object.TA_count;
        //this->dimension_count = object.dimension_count;
        for (int i = 0; i < 100; i++)
        {
            this->dimension[i] = object.dimension[i];
        }
        return *this;
    }
    //std::atomic_int measure;
    //std::atomic_int TA_count;
    int measure;
    int TA_count;
    //int dimension_count;
    char dimension[100];
    char* dimension_ptr[10];
};

class AggEdge
{
public:
    AggEdge();
    AggEdge(int src, int dst, int weight = 0, int TA_count = 0);
    AggEdge(const AggEdge& copy);
    AggEdge& operator=(const AggEdge& object)
    {
        this->src = object.src;
        this->dst = object.dst;
        this->weight = (int)object.weight;
        this->TA_count = (int)object.TA_count;
        return *this;
    }
    int src;
    int dst;
    //std::atomic_int weight;
    //std::atomic_int TA_count;
    int weight;
    int TA_count;
};

class CubeAggVertexValue  // element in verticesValue in Graph class
{
public:
    CubeAggVertexValue();
    CubeAggVertexValue(char dimension[100], int measure);
    int measure;
    //int dimension_count;
    char dimension[100];
    char* dimension_ptr[10];

    static const int max_dimension_count = 10;
    static const int max_dimesnion_len = 9;
};

class CubeAggGraph
{
public:
    CubeAggGraph();
    int vCount;
    int eCount;
    std::vector<Vertex> vList;
    std::vector<Edge> eList;
    std::vector<CubeAggVertexValue> verticesValue;
};

template <typename VertexValueType, typename MessageValueType>
class CubeAgg : public GraphUtil<VertexValueType, MessageValueType>
{
public:
    CubeAgg();

    int MSGApply(Graph<VertexValueType>& g, const std::vector<int>& initVSet, std::set<int>& activeVertice, const MessageSet<MessageValueType>& mSet) override;
    int MSGGenMerge(const Graph<VertexValueType>& g, const std::vector<int>& initVSet, const std::set<int>& activeVertice, MessageSet<MessageValueType>& mSet) override;

    //For transportation between jni part and processing part by using share memory
    //Also for less data transformation in order to achieve higher performance
    //Data struct Graph is not necessary!?
    int MSGApply_array(int vCount, int eCount, Vertex* vSet, int numOfInitV, const int* initVSet, VertexValueType* vValues, MessageValueType* mValues) override;
    int MSGGenMerge_array(int vCount, int eCount, const Vertex* vSet, const Edge* eSet, int numOfInitV, const int* initVSet, const VertexValueType* vValues, MessageValueType* mValues) override;

    //Master function
    void Init(int vCount, int eCount, int numOfInitV) override;
    void GraphInit(Graph<VertexValueType>& g, std::set<int>& activeVertices, const std::vector<int>& initVList) override;
    void Free() override;
    void Deploy(int vCount, int eCount, int numOfInitV) override;
    void MergeGraph(Graph<VertexValueType>& g, const std::vector<Graph<VertexValueType>>& subGSet,
        std::set<int>& activeVertices, const std::vector<std::set<int>>& activeVerticeSet,
        const std::vector<int>& initVList) override;

    // = = = = = = = = = = CubeAgg = = = = = = = = = =

    static void VC_threadBlock(int tid, int threadNum, CubeAggGraph& sub_g, std::vector<AggVertex>& aggVList,
        std::vector<int> aggDimension, long long& maxTimeCompute, long long& maxTimeMerge, int& mergeCycle);

    static void EC_threadBlock(int tid, int threadNum, CubeAggGraph& sub_g, std::vector<AggEdge>& aggEList,
        std::vector<int>& v2v, long long& maxTimeCompute, long long& maxTimeMerge, int& mergeCycle);

    void ApplyD(CubeAggGraph& g, const std::vector<std::vector<int>>& aggDimension,
        std::vector<CubeAggGraph>& agg_gSet, const std::vector<int>& dgList, int cuboidID);

    virtual void VertexCompute(std::vector<CubeAggGraph>& subGSet, std::vector<std::vector<AggVertex>>& aggVListSet,
        std::vector<int> aggDimension);

    virtual void EdgeCompute(std::vector<CubeAggGraph>& subGSet, std::vector<std::vector<AggEdge>>& aggEListSet);

    virtual void VertexMerge(CubeAggGraph& g, std::vector<std::vector<AggVertex>>& aggVListSet,
        std::vector<int> aggDimension);

    virtual void EdgeMerge(CubeAggGraph& g, std::vector<std::vector<AggEdge>>& aggEListSet);

    void AggInit();

    void AggEnd();

    void GetV2V(CubeAggGraph& g, std::vector<int> aggDimension);

    static std::string GetHashKeyAggVertex(const std::vector<int>& aggDimension, const char dimension[100], char* const dimension_ptr[10]);

    static std::string GetHashKeyMergeVertex(const char dimension[100], char* const dimension_ptr[10]);

    static long long GetHashKeyAggEdge(const std::vector<int>& v2v, int src, int dst);

    static long long GetHashKeyMergeEdge(int src, int dst);

    std::vector<CubeAggGraph> DivideGraph(CubeAggGraph& g, int partitionCount,
        int threadCount, std::vector<int> aggDimension);

    int GetMaxValueOfMap(const std::unordered_map<long long int, int>& mp);

    void GetOptimalMergeCycle(const CubeAggGraph& g, std::vector<int> aggDimension, int partitionNum, int threadNum);

    std::unordered_map<std::string, int> map_d2v;
    std::vector<int> v2v;

    long long time_VC_C;
    long long time_VC_M;
    long long time_VM;
    long long time_EC_C;
    long long time_EC_M;
    long long time_EM;
    long long time;

    long long fulltime_VC_C;
    long long fulltime_VC_M;
    long long fulltime_VM;
    long long fulltime_EC_C;
    long long fulltime_EC_M;
    long long fulltime_EM;
    long long fulltime;

    int currentPartition;   // current partition number
    int currentThread;      // current thread number for each partition
    int dCount;

    int mergeCycle_vertex;
    int mergeCycle_edge;

    int sumMergeCycle_vertex;
    int sumMergeCycle_edge;
    int sumComputeCycle_vertex;
    int sumComputeCycle_edge;

    int totalVValuesCount;  // vValue shared memory size
    int totalMValuesCount;  // vValue shared memory size
};

void GraphEntityPackingVertex(CubeAggGraph& g, int threadNum, std::vector<int> aggDimension,
    std::vector<std::vector<int>>& matrix_vertex);

void GraphEntityPackingVertexOrigin(CubeAggGraph& g, int threadNum, std::vector<int> aggDimension,
    std::vector<std::vector<int>>& matrix_vertex);

void GraphEntityPackingEdge(CubeAggGraph& g, int threadNum, std::vector<int>& v2v,
    std::vector<std::vector<int>>& matrix_edge);

void GraphEntityPackingEdgeOrigin(CubeAggGraph& g, int threadNum, std::vector<int>& v2v,
    std::vector<std::vector<int>>& matrix_edge);

int GetMergeCycleVertexPacking(CubeAggGraph& g, int threadNum, std::vector<int> aggDimension,
    std::vector<std::vector<int>>& matrix_vertex);

int GetMergeCycleEdgePacking(CubeAggGraph& g, int threadNum, std::vector<int>& v2v,
    std::vector<std::vector<int>>& matrix_edge);

long long GetTime(struct timeval end, struct timeval start);


#endif // !CUBEAGG_H

