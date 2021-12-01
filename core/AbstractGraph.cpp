//
// Created by Thoh Testarossa on 2019-07-19.
//

#include "../core/AbstractGraph.h"

Vertex::Vertex()
{
    this->inDegree = 0;
    this->outDegree = 0;
}

Vertex::Vertex(int vertexID, bool activeness, int initVIndex)
{
    this->vertexID = vertexID;
    this->isActive = activeness;
    this->initVIndex = initVIndex;
    this->inDegree = 0;
    this->outDegree = 0;
}

Vertex::Vertex(int vertexID, bool activeness, int initVIndex, int inDegree, int outDegree)
{
    this->vertexID = vertexID;
    this->isActive = activeness;
    this->initVIndex = initVIndex;
    this->inDegree = inDegree;
    this->outDegree = outDegree;
}

Edge::Edge()
{
    this->src = -1;
    this->dst = -1;
    this->weight = 0;
}

Edge::Edge(int src, int dst, double weight)
{
    this->src = src;
    this->dst = dst;
    this->weight = weight;
}

AbstractGraph::AbstractGraph(int vCount)
{
    this->vList = std::vector<Vertex>();
    this->eList = std::vector<Edge>();

    this->vCount = vCount;
    for(int i = 0; i < vCount; i++) this->vList.emplace_back(i, false, INVALID_INITV_INDEX);
    this->eCount = 0;
}

AbstractGraph::AbstractGraph(const std::vector<Vertex> &vSet, const std::vector<Edge> &eSet)
{
    this->vCount = vSet.size();
    this->eCount = eSet.size();
    this->vList = vSet;
    this->eList = eSet;
}

AbstractGraph::AbstractGraph(int vCount, int eCount, int *eSrcSet, int *eDstSet, double *eWeightSet)
{
    this->vCount = vCount;
    this->eCount = eCount;

    this->vList = std::vector<Vertex>();
    this->eList = std::vector<Edge>();

    //v assemble
    for(int i = 0; i < this->vCount; i++)
    {
        auto v = Vertex(i, false, INVALID_INITV_INDEX);
        this->vList.emplace_back(v);
    }

    //e assemble
    for(int i = 0; i < this->eCount; i++)
        this->eList.emplace_back(eSrcSet[i], eDstSet[i], eWeightSet[i]);
}

void AbstractGraph::insertEdge(int src, int dst, double weight)
{
    this->eList.emplace_back(src, dst, weight);
    this->eCount++;
}
