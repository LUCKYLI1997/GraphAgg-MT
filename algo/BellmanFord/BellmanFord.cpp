//
// Created by Thoh Testarossa on 2019-03-08.
//

#include "BellmanFord.h"

#include <iostream>
#include <ctime>

template <typename VertexValueType, typename MessageValueType>
BellmanFord<VertexValueType, MessageValueType>::BellmanFord()
{
}

template <typename VertexValueType, typename MessageValueType>
int BellmanFord<VertexValueType, MessageValueType>::MSGApply(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertice, const MessageSet<MessageValueType> &mSet)
{
    //Activity reset
    activeVertice.clear();

    //Availability check
    if(g.vCount <= 0) return 0;

    //MSG Init
    MessageValueType *mValues = new MessageValueType [g.vCount * this->numOfInitV];

    for(int i = 0; i < g.vCount * this->numOfInitV; i++)
        mValues[i] = (MessageValueType)INVALID_MASSAGE;
    for(int i = 0; i < mSet.mSet.size(); i++)
    {
        auto &mv = mValues[mSet.mSet.at(i).dst * this->numOfInitV + g.vList.at(mSet.mSet.at(i).src).initVIndex];
        if(mv > mSet.mSet.at(i).value)
            mv = mSet.mSet.at(i).value;
    }

    //array form computation
    this->MSGApply_array(g.vCount, g.eCount, &g.vList[0], this->numOfInitV, &initVSet[0], &g.verticesValue[0], mValues);

    //Active vertices set assembly
    for(int i = 0; i < g.vCount; i++)
    {
        if(g.vList.at(i).isActive)
            activeVertice.insert(i);
    }

    free(mValues);

    return activeVertice.size();
}

template <typename VertexValueType, typename MessageValueType>
int BellmanFord<VertexValueType, MessageValueType>::MSGGenMerge(const Graph<VertexValueType> &g, const std::vector<int> &initVSet, const std::set<int> &activeVertice, MessageSet<MessageValueType> &mSet)
{
    //Generate merged msgs directly

    //Availability check
    if(g.vCount <= 0) return 0;

    //mValues init
    MessageValueType *mValues = new MessageValueType [g.vCount * this->numOfInitV];

    //array form computation
    this->MSGGenMerge_array(g.vCount, g.eCount, &g.vList[0], &g.eList[0], this->numOfInitV, &initVSet[0], &g.verticesValue[0], mValues);

    //Package mMergedMSGValueSet to result mSet
    for(int i = 0; i < g.vCount * this->numOfInitV; i++)
    {
        if(mValues[i] != (MessageValueType)INVALID_MASSAGE)
        {
            int dst = i / this->numOfInitV;
            int initV = initVSet[i % this->numOfInitV];
            mSet.insertMsg(Message<MessageValueType>(initV, dst, mValues[i]));
        }
    }

    free(mValues);

    return mSet.mSet.size();
}

template <typename VertexValueType, typename MessageValueType>
int BellmanFord<VertexValueType, MessageValueType>::MSGApply_array(int vCount, int eCount, Vertex *vSet, int numOfInitV, const int *initVSet, VertexValueType *vValues, MessageValueType *mValues)
{
    int avCount = 0;

    for(int i = 0; i < vCount; i++) vSet[i].isActive = false;

    for(int i = 0; i < vCount * numOfInitV; i++)
    {
        if(vValues[i] > (VertexValueType)mValues[i])
        {
            vValues[i] = (VertexValueType)mValues[i];
            if(!vSet[i / numOfInitV].isActive)
            {
                vSet[i / numOfInitV].isActive = true;
                avCount++;
            }
        }
    }

    return avCount;
}

template <typename VertexValueType, typename MessageValueType>
int BellmanFord<VertexValueType, MessageValueType>::MSGGenMerge_array(int vCount, int eCount, const Vertex *vSet, const Edge *eSet, int numOfInitV, const int *initVSet, const VertexValueType *vValues, MessageValueType *mValues)
{
    ////test
    //std::cout << (long long)vSet << std::endl;

    for(int i = 0; i < vCount * numOfInitV; i++) mValues[i] = (MessageValueType)INVALID_MASSAGE;

    for(int i = 0; i < eCount; i++)
    {
        if(vSet[eSet[i].src].isActive)
        {
            for(int j = 0; j < numOfInitV; j++)
            {
                if(mValues[eSet[i].dst * numOfInitV + j] > (MessageValueType)vValues[eSet[i].src * numOfInitV + j] + eSet[i].weight)
                    mValues[eSet[i].dst * numOfInitV + j] = (MessageValueType)vValues[eSet[i].src * numOfInitV + j] + eSet[i].weight;
            }
        }
    }

    return vCount * numOfInitV;
}

template <typename VertexValueType, typename MessageValueType>
void BellmanFord<VertexValueType, MessageValueType>::Init(int vCount, int eCount, int numOfInitV)
{
    this->numOfInitV = numOfInitV;

    //Memory parameter init
    this->totalVValuesCount = vCount * numOfInitV;
    this->totalMValuesCount = vCount * numOfInitV;
}

template <typename VertexValueType, typename MessageValueType>
void BellmanFord<VertexValueType, MessageValueType>::GraphInit(Graph<VertexValueType> &g, std::set<int> &activeVertices,
                                             const std::vector<int> &initVList)
{
    int numOfInitV_init = initVList.size();

    //v Init
    for(int i = 0; i < numOfInitV_init; i++)
        g.vList.at(initVList.at(i)).initVIndex = i;
    for(auto &v : g.vList)
    {
        if(v.initVIndex != INVALID_INITV_INDEX)
        {
            activeVertices.insert(v.vertexID);
            v.isActive = true;
        }
        else v.isActive = false;
    }

    //vValues init
    g.verticesValue.reserve(g.vCount * numOfInitV_init);
    g.verticesValue.assign(g.vCount * numOfInitV_init, (VertexValueType)(INT32_MAX >> 1));
    for(int initID : initVList)
        g.verticesValue.at(initID * numOfInitV_init + g.vList.at(initID).initVIndex) = (VertexValueType)0;
}

template <typename VertexValueType, typename MessageValueType>
void BellmanFord<VertexValueType, MessageValueType>::Deploy(int vCount, int eCount, int numOfInitV)
{

}

template <typename VertexValueType, typename MessageValueType>
void BellmanFord<VertexValueType, MessageValueType>::Free()
{

}

template <typename VertexValueType, typename MessageValueType>
void BellmanFord<VertexValueType, MessageValueType>::MergeGraph(Graph<VertexValueType> &g, const std::vector<Graph<VertexValueType>> &subGSet,
                std::set<int> &activeVertices, const std::vector<std::set<int>> &activeVerticeSet,
                const std::vector<int> &initVList)
{
    //Init
    activeVertices.clear();
    for(auto &v : g.vList) v.isActive = false;

    //Merge graphs
    for(const auto &subG : subGSet)
    {
        //vSet merge
        for(int i = 0; i < subG.vCount; i++)
            g.vList.at(i).isActive |= subG.vList.at(i).isActive;

        //vValues merge
        for(int i = 0; i < subG.verticesValue.size(); i++)
        {
            if(g.verticesValue.at(i) > subG.verticesValue.at(i))
                g.verticesValue.at(i) = subG.verticesValue.at(i);
        }
    }

    //Merge active vertices set
    for(const auto &AVs : activeVerticeSet)
    {
        for(auto av : AVs)
            activeVertices.insert(av);
    }
}

template <typename VertexValueType, typename MessageValueType>
void BellmanFord<VertexValueType, MessageValueType>::ApplyStep(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertices)
{
    auto mGenSet = MessageSet<MessageValueType>();
    auto mMergedSet = MessageSet<MessageValueType>();

    mMergedSet.mSet.clear();
    MSGGenMerge(g, initVSet, activeVertices, mMergedSet);

    //Test
    std::cout << "MGenMerge:" << clock() << std::endl;
    //Test end

    activeVertices.clear();
    MSGApply(g, initVSet, activeVertices, mMergedSet);

    //Test
    std::cout << "Apply:" << clock() << std::endl;
    //Test end
}

template <typename VertexValueType, typename MessageValueType>
void BellmanFord<VertexValueType, MessageValueType>::Apply(Graph<VertexValueType> &g, const std::vector<int> &initVList)
{
    //Init the Graph
    std::set<int> activeVertices = std::set<int>();
    auto mGenSet = MessageSet<MessageValueType>();
    auto mMergedSet = MessageSet<MessageValueType>();

    Init(g.vCount, g.eCount, initVList.size());

    GraphInit(g, activeVertices, initVList);

    Deploy(g.vCount, g.eCount, initVList.size());

    while(activeVertices.size() > 0)
        ApplyStep(g, initVList, activeVertices);

    Free();
}

template <typename VertexValueType, typename MessageValueType>
void BellmanFord<VertexValueType, MessageValueType>::ApplyD(Graph<VertexValueType> &g, const std::vector<int> &initVList, int partitionCount)
{
    //Init the Graph
    std::set<int> activeVertices = std::set<int>();

    std::vector<std::set<int>> AVSet = std::vector<std::set<int>>();
    for(int i = 0; i < partitionCount; i++) AVSet.push_back(std::set<int>());
    auto mGenSetSet = std::vector<MessageSet<MessageValueType>>();
    for(int i = 0; i < partitionCount; i++) mGenSetSet.push_back(MessageSet<MessageValueType>());
    auto mMergedSetSet = std::vector<MessageSet<MessageValueType>>();
    for(int i = 0; i < partitionCount; i++) mMergedSetSet.push_back(MessageSet<MessageValueType>());

    Init(g.vCount, g.eCount, initVList.size());

    GraphInit(g, activeVertices, initVList);

    Deploy(g.vCount, g.eCount, initVList.size());

    int iterCount = 0;

    while(activeVertices.size() > 0)
    {
        //Test
        std::cout << ++iterCount << ":" << clock() << std::endl;
        //Test end

        auto subGraphSet = this->DivideGraphByEdge(g, partitionCount);

        for(int i = 0; i < partitionCount; i++)
        {
            AVSet.at(i).clear();
            AVSet.at(i) = activeVertices;
        }

        //Test
        std::cout << "GDivide:" << clock() << std::endl;
        //Test end

        for(int i = 0; i < partitionCount; i++)
            ApplyStep(subGraphSet.at(i), initVList, AVSet.at(i));

        activeVertices.clear();
        MergeGraph(g, subGraphSet, activeVertices, AVSet, initVList);
        //Test
        std::cout << "GMerge:" << clock() << std::endl;
        //Test end
    }

    Free();

    //Test
    std::cout << "end" << ":" << clock() << std::endl;
    //Test end
}
