//
// Created by Thoh Testarossa on 2019-03-08.
//

#include "../core/Graph.h"

template <typename VertexValueType>
Graph<VertexValueType>::Graph(int vCount) : AbstractGraph(vCount)
{
    this->verticesValue = std::vector<VertexValueType>();
}

template <typename VertexValueType>
Graph<VertexValueType>::Graph(const std::vector<Vertex> &vSet, const std::vector<Edge> &eSet, const std::vector<VertexValueType> &verticesValue) : AbstractGraph(vSet, eSet)
{
    this->verticesValue = verticesValue;
}

template<typename VertexValueType>
Graph<VertexValueType>::Graph(int vCount, int eCount, int *eSrcSet, int *eDstSet, double *eWeightSet, bool *AVCheckSet) : AbstractGraph(vCount, eCount, eSrcSet, eDstSet, eWeightSet)
{
    this->verticesValue = std::vector<VertexValueType>();

    //AVCheck init
    for(int i = 0; i < vCount; i++)
        this->vList.at(i).isActive = AVCheckSet[i];
}

template <typename VertexValueType>
Graph<VertexValueType>::Graph(int vCount, int eCount, int numOfInitV, const int *initVSet, int *eSrcSet, int *eDstSet, double *eWeightSet, bool *AVCheckSet) : AbstractGraph(vCount, eCount, eSrcSet, eDstSet, eWeightSet)
{
    this->verticesValue = std::vector<VertexValueType>();

    //AVCheck init
    for(int i = 0; i < vCount; i++)
        this->vList.at(i).isActive = AVCheckSet[i];

    //initVIndex will be initialized after other initializations finished
    for(int i = 0; i < numOfInitV; i++)
        this->vList.at(initVSet[i]).initVIndex = i;
}
