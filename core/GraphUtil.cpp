//
// Created by Thoh Testarossa on 2019-03-11.
//

#include "GraphUtil.h"

template <typename VertexValueType, typename MessageValueType>
std::vector<Graph<VertexValueType>> GraphUtil<VertexValueType, MessageValueType>::DivideGraphByEdge(const Graph<VertexValueType> &g, int partitionCount)
{
    std::vector<Graph<VertexValueType>> res = std::vector<Graph<VertexValueType>>();
    for(int i = 0; i < partitionCount; i++) res.push_back(Graph<VertexValueType>(0));
    for(int i = 0; i < partitionCount; i++)
    {
        //Copy v & vValues info but do not copy e info
        res.at(i) = Graph<VertexValueType>(g.vList, std::vector<Edge>(), g.verticesValue);

        //Distribute e info
        for(int k = i * g.eCount / partitionCount; k < (i + 1) * g.eCount / partitionCount; k++)
            res.at(i).insertEdge(g.eList.at(k).src, g.eList.at(k).dst, g.eList.at(k).weight);
    }

    return res;
}

template <typename VertexValueType, typename MessageValueType>
int GraphUtil<VertexValueType, MessageValueType>::reflect(const std::vector<int> &originalIntList, int originalIntRange, std::vector<int> &reflectIndex, std::vector<int> &reversedIndex)
{
    //Init
    //Guarantee: size of reflectIndex is not greater than originalIntList.size(), and size of reversedIndex is not greater than originalIntRange
    reflectIndex.reserve(originalIntList.size());
    reversedIndex.reserve(originalIntRange);
    reversedIndex.assign(originalIntRange, NO_REFLECTION);

    //Reflection
    int reflectCount = 0;

    for(auto o_i : originalIntList)
    {
        if(reversedIndex.at(o_i) == NO_REFLECTION)
        {
            reflectIndex.emplace_back(o_i);
            reversedIndex.at(o_i) = reflectCount++;
        }
    }

    return reflectCount;
}

template <typename VertexValueType, typename MessageValueType>
Graph<VertexValueType>
GraphUtil<VertexValueType, MessageValueType>::reflectG(const Graph<VertexValueType> &o_g, const std::vector<Edge> &eSet, std::vector<int> &reflectIndex, std::vector<int> &reversedIndex)
{
    //Init
    int vCount = o_g.vCount;
    int eCount = eSet.size();

    reflectIndex.clear();
    reversedIndex.clear();
    reflectIndex.reserve(2 * eCount);
    reversedIndex.reserve(vCount);

    //Calculate reflection using eSet and generate reflected eSet
    auto r_eSet = std::vector<Edge>();
    r_eSet.reserve(2 * eCount);

    auto originalIntList = std::vector<int>();
    originalIntList.reserve(2 * eCount);

    for(const auto &e : eSet)
    {
        originalIntList.emplace_back(e.src);
        originalIntList.emplace_back(e.dst);
    }

    int reflectCount = this->reflect(originalIntList, vCount, reflectIndex, reversedIndex);

    //Generate reflected eSet
    for(const auto &e : eSet)
        r_eSet.emplace_back(reversedIndex.at(e.src), reversedIndex.at(e.dst), e.weight);

    //Generate reflected vSet & vValueSet
    auto r_vSet = std::vector<Vertex>();
    r_vSet.reserve(reflectCount * sizeof(Vertex));

    int numOfInitV = o_g.verticesValue.size() / o_g.vCount;
    auto r_vValueSet = std::vector<VertexValueType>();
    r_vValueSet.reserve(reflectCount * numOfInitV * sizeof(VertexValueType));

    for(int i = 0; i < reflectCount; i++)
    {
        r_vSet.emplace_back(o_g.vList.at(reflectIndex.at(i)));
        for(int j = 0; j < numOfInitV; j++)
            r_vValueSet.emplace_back(o_g.verticesValue.at(reflectIndex.at(i) * numOfInitV + j));

        r_vSet.at(i).vertexID = i;
    }

    //Generate reflected graph and return
    return Graph<VertexValueType>(r_vSet, r_eSet, r_vValueSet);
}

template <typename VertexValueType, typename MessageValueType>
MessageSet<MessageValueType>
GraphUtil<VertexValueType, MessageValueType>::reflectM(const MessageSet<MessageValueType> &o_mSet, int vCount, std::vector<int> &reflectIndex, std::vector<int> &reversedIndex)
{
    auto r_mSet = MessageSet<MessageValueType>();

    reflectIndex.reserve(o_mSet.mSet.size());
    reversedIndex.reserve(vCount);
    reversedIndex.assign(vCount, NO_REFLECTION);

    auto originalIntList = std::vector<int>();
    originalIntList.reserve(o_mSet.mSet.size());

    for(const auto &m : o_mSet.mSet) originalIntList.emplace_back(m.dst);

    int reflectCount = this->reflect(originalIntList, vCount, reflectIndex, reversedIndex);

    for(const auto &m : o_mSet.mSet) r_mSet.insertMsg(Message<MessageValueType>(m.src, reversedIndex.at(m.dst), m.value));

    return r_mSet;
}
