//
// Created by cave-g-f on 2019-05-17.
//

#include "../algo/LabelPropagation/LabelPropagation.h"

#include <iostream>
#include <chrono>

template <typename VertexValueType, typename MessageValueType>
LabelPropagation<VertexValueType, MessageValueType>::LabelPropagation()
{
}

template <typename VertexValueType, typename MessageValueType>
int LabelPropagation<VertexValueType, MessageValueType>::MSGApply(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertices, const MessageSet<MessageValueType> &mSet)
{
    //store the labels of neighbors for each vertice
    auto labelsVector = std::vector<std::map<int, int>>();

    //store the most often label for each vertice
    auto maxCntLabel = std::vector<std::pair<int, int>>();

    labelsVector.reserve(g.vCount);
    labelsVector.assign(g.vCount, std::map<int, int>());
    maxCntLabel.reserve(g.vCount);
    maxCntLabel.assign(g.vCount, std::pair<int, int>(0, 0));

    for(const auto &m : mSet.mSet)
    {
        auto &labelCntMap = labelsVector.at(m.dst);
        auto &maxLabel = maxCntLabel.at(m.dst);
        
        if(labelCntMap.find(m.value.second) == labelCntMap.end())
            labelCntMap.insert(std::map<int, int>::value_type(m.value.second, 1));
        else
            labelCntMap.at(m.value.second)++;
        
        if(maxLabel.second < labelCntMap.at(m.value.second))
        {
            maxLabel.first = m.value.second;
            maxLabel.second = labelCntMap.at(m.value.second);
        }
    }

    for(int i = 0; i < g.vList.size(); i++)
    {
        auto label = maxCntLabel.at(i);

        if(label.second != 0)
        {
            g.verticesValue.at(i).first = label.first;
            g.verticesValue.at(i).second = label.second;
        }
    }

    return 0;
}

template <typename VertexValueType, typename MessageValueType>
int LabelPropagation<VertexValueType, MessageValueType>::MSGGenMerge(const Graph<VertexValueType> &g, const std::vector<int> &initVSet, const std::set<int> &activeVertice, MessageSet<MessageValueType> &mSet)
{
    //Generate merged msgs directly
    mSet.mSet.clear();
    mSet.mSet.reserve(g.eCount);

    for(auto e : g.eList)
    {
        //msg value -- <destinationID, Label>
        auto msgValue = std::pair<int, int>(e.dst, g.verticesValue.at(e.src).first);
        mSet.insertMsg(Message<MessageValueType>(e.src, e.dst, msgValue));
    }

    return mSet.mSet.size();
}

template <typename VertexValueType, typename MessageValueType>
int LabelPropagation<VertexValueType, MessageValueType>::MSGApply_array(int vCount, int eCount, Vertex *vSet, int numOfInitV, const int *initVSet, VertexValueType *vValues, MessageValueType *mValues)
{
    //store the labels of neighbors for each vertice
    auto labelsVector = std::vector<std::map<int, int>>();

    //store the most often label for each vertice
    auto maxCntLabel = std::vector<std::pair<int, int>>();

    labelsVector.reserve(vCount);
    labelsVector.assign(vCount, std::map<int, int>());
    maxCntLabel.reserve(vCount);
    maxCntLabel.assign(vCount, std::pair<int, int>(0, 0));

    for(int i = 0; i < eCount; i++)
    {
        auto &labelCntMap = labelsVector.at(mValues[i].first);
        auto &maxLabel = maxCntLabel.at(mValues[i].first);

        if(labelCntMap.find(mValues[i].second) == labelCntMap.end())
        {
            labelCntMap.insert(std::map<int, int>::value_type(mValues[i].second, 1));
        }
        else
        {
            labelCntMap.at(mValues[i].second)++;
        }
        
        if(maxLabel.second < labelCntMap.at(mValues[i].second))
        {
            maxLabel.first = mValues[i].second;
            maxLabel.second = labelCntMap.at(mValues[i].second);
        }
    }

    for(int i = 0; i < vCount; i++)
    {
        auto label = maxCntLabel.at(i);

        if(label.second != 0)
        {
           vValues[i].first = label.first;
           vValues[i].second = label.second;
        }
    }

    return 0;
}

template <typename VertexValueType, typename MessageValueType>
int LabelPropagation<VertexValueType, MessageValueType>::MSGGenMerge_array(int vCount, int eCount, const Vertex *vSet, const Edge *eSet, int numOfInitV, const int *initVSet, const VertexValueType *vValues, MessageValueType *mValues)
{
    for(int i = 0; i < eCount; i++)
    {
        //msg value -- <destinationID, Label>
        auto msgValue = std::pair<int, int>(eSet[i].dst, vValues[eSet[i].src].first);
        mValues[i] = msgValue;
    }

    return eCount;
}

template <typename VertexValueType, typename MessageValueType>
void LabelPropagation<VertexValueType, MessageValueType>::Init(int vCount, int eCount, int numOfInitV)
{
    this->totalVValuesCount = vCount;
    this->totalMValuesCount = eCount;
}

template <typename VertexValueType, typename MessageValueType>
void LabelPropagation<VertexValueType, MessageValueType>::GraphInit(Graph<VertexValueType> &g, std::set<int> &activeVertices, const std::vector<int> &initVList)
{
    //vValues init
    g.verticesValue.reserve(g.vCount);
    g.verticesValue.assign(g.vCount, std::pair<int, int>(0, 0));

    for(int i = 0; i < g.vList.size(); i++)
        g.verticesValue.at(i).first = g.vList.at(i).vertexID;
}

template <typename VertexValueType, typename MessageValueType>
void LabelPropagation<VertexValueType, MessageValueType>::Deploy(int vCount, int eCount, int numOfInitV)
{

}

template <typename VertexValueType, typename MessageValueType>
void LabelPropagation<VertexValueType, MessageValueType>::Free()
{

}

template <typename VertexValueType, typename MessageValueType>
void LabelPropagation<VertexValueType, MessageValueType>::MergeGraph(Graph<VertexValueType> &g, const std::vector<Graph<VertexValueType>> &subGSet,
                    std::set<int> &activeVertices, const std::vector<std::set<int>> &activeVerticeSet,
                    const std::vector<int> &initVList)
{
    //Merge graphs
    auto resG = Graph<VertexValueType>(0);

    if(subGSet.size() <= 0);
    else
    {
        resG = subGSet.at(0);

        resG.eList.clear();
        resG.eCount = 0;

        for(const auto &subG : subGSet) resG.eCount += subG.eCount;
        resG.eList.reserve(resG.eCount);

        //Merge subGraphs
        for(const auto &subG : subGSet)
        {
            //Merge vValues
            for(int i = 0; i < subG.verticesValue.size(); i++)
            {
                if(resG.verticesValue.at(i).first != subG.verticesValue.at(i).first && resG.verticesValue.at(i).second < subG.verticesValue.at(i).second)
                {
                    resG.verticesValue.at(i).first = subG.verticesValue.at(i).first;
                    resG.verticesValue.at(i).second = subG.verticesValue.at(i).second;
                }
            }

            //Merge Edge
            //There should be not any relevant edges since subgraphs are divided by dividing edge set
            resG.eList.insert(resG.eList.end(), subG.eList.begin(), subG.eList.end());
        }
    }

    g = resG;
}

template <typename VertexValueType, typename MessageValueType>
void LabelPropagation<VertexValueType, MessageValueType>::ApplyStep(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertices)
{
    MessageSet<MessageValueType> mMergedSet = MessageSet<MessageValueType>();

    mMergedSet.mSet.clear();

    auto start = std::chrono::system_clock::now();
    MSGGenMerge(g, initVSet, activeVertices, mMergedSet);
    auto mergeEnd = std::chrono::system_clock::now();

    MSGApply(g, initVSet, activeVertices, mMergedSet);
    auto applyEnd = std::chrono::system_clock::now();
}

template <typename VertexValueType, typename MessageValueType>
void LabelPropagation<VertexValueType, MessageValueType>::Apply(Graph<VertexValueType> &g, const std::vector<int> &initVList)
{
    //Init the Graph
    std::set<int> activeVertice = std::set<int>();
    MessageSet<MessageValueType> mGenSet = MessageSet<MessageValueType>();
    MessageSet<MessageValueType> mMergedSet = MessageSet<MessageValueType>();

    Init(g.vCount, g.eCount, initVList.size());

    GraphInit(g, activeVertice, initVList);

    Deploy(g.vCount, g.eCount, initVList.size());

    while(activeVertice.size() > 0)
        ApplyStep(g, initVList, activeVertice);

    Free();
}


template <typename VertexValueType, typename MessageValueType>
void LabelPropagation<VertexValueType, MessageValueType>::ApplyD(Graph<VertexValueType> &g, const std::vector<int> &initVList, int partitionCount)
{
    //Init the Graph
    std::set<int> activeVertice = std::set<int>();

    std::vector<std::set<int>> AVSet = std::vector<std::set<int>>();
    for(int i = 0; i < partitionCount; i++) AVSet.push_back(std::set<int>());
    std::vector<MessageSet<MessageValueType>> mGenSetSet = std::vector<MessageSet<MessageValueType>>();
    for(int i = 0; i < partitionCount; i++) mGenSetSet.push_back(MessageSet<MessageValueType>());
    std::vector<MessageSet<MessageValueType>> mMergedSetSet = std::vector<MessageSet<MessageValueType>>();
    for(int i = 0; i < partitionCount; i++) mMergedSetSet.push_back(MessageSet<MessageValueType>());

    Init(g.vCount, g.eCount, initVList.size());

    GraphInit(g, activeVertice, initVList);

    Deploy(g.vCount, g.eCount, initVList.size());

    int iterCount = 0;

    while(iterCount < 100)
    {
        //std::cout << "iterCount: " << iterCount << std::endl;
        auto start = std::chrono::system_clock::now();
        auto subGraphSet = this->DivideGraphByEdge(g, partitionCount);
        auto divideGraphFinish = std::chrono::system_clock::now();

        for(int i = 0; i < partitionCount; i++)
            ApplyStep(subGraphSet.at(i), initVList, AVSet.at(i));

        activeVertice.clear();

        auto mergeGraphStart = std::chrono::system_clock::now();
        MergeGraph(g, subGraphSet, activeVertice, AVSet, initVList);
        iterCount++;
        auto end = std::chrono::system_clock::now();
    } 

    for(int i = 0; i < g.vCount; i++)
        std::cout << g.verticesValue.at(i).first << std::endl;

    Free();
}
