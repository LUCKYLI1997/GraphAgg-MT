//
// Created by Thoh Testarossa on 2019-08-22.
//

#include "DDFS.h"

#include <iostream>
#include <ctime>
#include <queue>

template <typename VertexValueType, typename MessageValueType>
DDFS<VertexValueType, MessageValueType>::DDFS()
{

}

template <typename VertexValueType, typename MessageValueType>
int DDFS<VertexValueType, MessageValueType>::MSGApply(Graph<VertexValueType> &g, const std::vector<int> &initVSet,
                                     std::set<int> &activeVertice, const MessageSet<MessageValueType> &mSet)
{
    //Activity reset
    activeVertice.clear();

    //Availability check
    if(g.vCount <= 0) return 0;

    //Organize MessageValueType vector
    auto tmpMSGVector = std::vector<MessageValueType>();
    tmpMSGVector.reserve(mSet.mSet.size());
    for(const auto &m : mSet.mSet) tmpMSGVector.emplace_back(m.value);

    //array form computation
    this->MSGApply_array(g.vCount, mSet.mSet.size(), &g.vList[0], this->numOfInitV, &initVSet[0], &g.verticesValue[0], &tmpMSGVector[0]);

    //Active vertices set assembly
    for(int i = 0; i < g.vCount; i++)
    {
        if(g.vList.at(i).isActive)
            activeVertice.insert(i);
    }

    return activeVertice.size();
}

template <typename VertexValueType, typename MessageValueType>
int DDFS<VertexValueType, MessageValueType>::MSGGenMerge(const Graph<VertexValueType> &g, const std::vector<int> &initVSet,
                                        const std::set<int> &activeVertice, MessageSet<MessageValueType> &mSet)
{
    //Availability check
    if(g.vCount <= 0) return 0;

    //Reset mSet
    mSet.mSet.clear();
    mSet.mSet.reserve(2 * g.eCount);

    auto tmpMSGSet = std::vector<MessageValueType>(2 * g.eCount, MessageValueType());

    //array form computation
    this->MSGGenMerge_array(g.vCount, g.eCount, &g.vList[0], &g.eList[0], this->numOfInitV, &initVSet[0], &g.verticesValue[0], &tmpMSGSet[0]);

    //Package msgs
    for(const auto &m : tmpMSGSet)
    {
        if(m.src != -1) mSet.insertMsg(m);
        else break;
    }

    return mSet.mSet.size();
}

template <typename VertexValueType, typename MessageValueType>
int DDFS<VertexValueType, MessageValueType>::MSGApply_array(int vCount, int eCount, Vertex *vSet, int numOfInitV, const int *initVSet,
                                           VertexValueType *vValues, MessageValueType *mValues)
{
    int avCount = 0;

    //Reset vertex activity
    for(int i = 0; i < vCount; i++)
        vSet[i].isActive = false;

    //Reset opbit & vNextMSGTo
    for(int i = 0; i < vCount; i++)
    {
        vValues[i].opbit = 0;
        vValues[i].vNextMSGTo = -1;
    }

    //Check each msgs
    //msgs are sent from edges
    //eCount here is the account of edges which contains messages rather than the account of g's edges
    //(eCount = mValues.size)
    for(int i = 0; i < eCount; i++)
    {
        //msg token check
        if(mValues[i].msgbit & MSG_TOKEN)
        {
            if(vValues[mValues[i].dst].state == STATE_IDLE)
            {
                //Mark j as i's father
                //There should be some approach more efficient
                for(auto &vState : vValues[mValues[i].dst].vStateList)
                {
                    if(vState.first == mValues[i].src)
                    {
                        vState.second = MARK_PARENT;
                        break;
                    }
                }

                vValues[mValues[i].dst].state = STATE_DISCOVERED;
                vValues[mValues[i].dst].vNextMSGTo = this->search(mValues[i].dst, numOfInitV, initVSet, vSet, vValues, avCount);

                //prepare to broadcast msg "visited" to other vertices
                vValues[mValues[i].dst].opbit |= OP_BROADCAST;

                //Vertex which will send msg will be activated
                if(!vSet[mValues[i].dst].isActive)
                {
                    vSet[mValues[i].dst].isActive = true;
                    avCount++;
                }
            }
        }
        //msg visited check
        else if(mValues[i].msgbit & MSG_VISITED)
        {
            //There should be some approach more efficient
            for(auto &vState : vValues[mValues[i].dst].vStateList)
            {
                if(vState.first == mValues[i].src)
                {
                    if(vState.second == MARK_UNVISITED)
                    {
                        vState.second = MARK_VISITED;
                        vValues[mValues[i].dst].vNextMSGTo = -1;
                    }
                    else if(vState.second == MARK_SON)
                    {
                        vState.second = MARK_VISITED;
                        vValues[mValues[i].dst].vNextMSGTo = this->search(mValues[i].dst, numOfInitV, initVSet, vSet, vValues, avCount);
                    }
                }
            }
        }
        else;
    }

    return avCount;
}

template <typename VertexValueType, typename MessageValueType>
int
DDFS<VertexValueType, MessageValueType>::MSGGenMerge_array(int vCount, int eCount, const Vertex *vSet, const Edge *eSet, int numOfInitV,
                                         const int *initVSet, const VertexValueType *vValues, MessageValueType *mValues)
{
    int msgCount = 0;

    for(int i = 0; i < vCount; i++)
    {
        if(vSet[i].isActive)
        {
            //Check if needed to generate broadcast msg
            if (vValues[i].opbit & OP_BROADCAST)
            {
                for (const auto &vState : vValues[i].vStateList)
                {
                    if (vState.second == MARK_UNVISITED || vState.second == MARK_VISITED)
                    {
                        mValues[msgCount].src = i;
                        mValues[msgCount].dst = vState.first;
                        mValues[msgCount].msgbit = MSG_VISITED;
                        //Not implemented yet
                        mValues[msgCount].timestamp = 0;

                        msgCount++;
                    }
                }
            }
            //Check if needed to generate search msg
            if (vValues[i].opbit & OP_MSG_FROM_SEARCH)
            {
                mValues[msgCount].src = i;
                mValues[msgCount].dst = vValues[i].vNextMSGTo;
                mValues[msgCount].msgbit = MSG_TOKEN;
                //Not implemented yet
                mValues[msgCount].timestamp = 0;

                msgCount++;
            }
        }
    }

    return msgCount;
}

template <typename VertexValueType, typename MessageValueType>
void DDFS<VertexValueType, MessageValueType>::MergeGraph(Graph<VertexValueType> &g, const std::vector<Graph<VertexValueType>> &subGSet,
                                       std::set<int> &activeVertices,
                                       const std::vector<std::set<int>> &activeVerticeSet,
                                       const std::vector<int> &initVList)
{
    //Reset global vValues
    for(auto &vV : g.verticesValue)
    {
        //state reset
        vV.state = STATE_IDLE;
        //vNextMSGTo reset
        vV.vNextMSGTo = -1;
        //opbit reset
        vV.opbit = (char)0;

        //Didn't be implemented yet
        //startTime reset
        //endTime reset
    }

    //Merge subGs parameters
    for(const auto &subG : subGSet)
    {
        for(int i = 0; i < g.vCount; i++)
        {
            const auto &vVSub = subG.verticesValue.at(i);
            auto &vV = g.verticesValue.at(i);
            //state merge
            vV.state |= vVSub.state;
            //vNextMSGTo merge
            if(!(vV.opbit & OP_MSG_FROM_SEARCH) && (vVSub.opbit & OP_MSG_FROM_SEARCH)) vV.vNextMSGTo = vVSub.vNextMSGTo;
            else if(!(vV.opbit & OP_MSG_DOWNWARD) && (vVSub.opbit & OP_MSG_DOWNWARD)) vV.vNextMSGTo = vVSub.vNextMSGTo;
            else;
            //opbit merge
            vV.opbit |= vVSub.opbit;

            //Didn't be implemented yet
            //startTime merge
            //endTime merge
        }
    }

    //Merge subG vStateList
    int subGCount = subGSet.size();
    int *subGIndex = new int [subGCount];
    for(int i = 0; i < subGCount; i++) subGIndex[i] = 0;

    for(int i = 0; i < g.vCount; i++)
    {
        auto &vV = g.verticesValue.at(i);
        for(int j = 0; j < vV.relatedVCount; j++)
        {
            for(int k = 0; k < subGCount; k++)
            {
                if(vV.vStateList.at(j).first == subGSet.at(k).verticesValue.at(i).vStateList.at(subGIndex[k]).first)
                {
                    vV.vStateList.at(j).second = subGSet.at(k).verticesValue.at(i).vStateList.at(subGIndex[k]).second;
                    subGIndex[k]++;
                    break;
                }
            }
        }
    }

    //Merge activeVertices
    activeVertices.clear();

    for(const auto &avs : activeVerticeSet)
    {
        for(const auto &av : avs)
            activeVertices.insert(av);
    }
}

template <typename VertexValueType, typename MessageValueType>
void DDFS<VertexValueType, MessageValueType>::Init(int vCount, int eCount, int numOfInitV)
{
    this->numOfInitV = numOfInitV;

    //Memory parameter init
    this->totalVValuesCount = vCount;
    this->totalMValuesCount = eCount;
}

template <typename VertexValueType, typename MessageValueType>
void DDFS<VertexValueType, MessageValueType>::GraphInit(Graph<VertexValueType> &g, std::set<int> &activeVertices,
                                      const std::vector<int> &initVList)
{
    int avCount = 0;

    //Global init
    //Init graph parameters
    for(int i = 0; i < g.vCount; i++) g.verticesValue.emplace_back(VertexValueType());
    //Scan edges in graph and collect info
    /*
     * for edge (a, b):
     *     add pair (b, MARK_UNVISITED) as (vid, mark) into a's vState priority queue ordered by vid
     *     add pair (a, MARK_UNVISITED) as (vid, mark) into b's vState priority queue ordered by vid
    */
    auto pqVector = std::vector<std::priority_queue<std::pair<int, char>, std::vector<std::pair<int, char>>, cmp>>(g.vCount, std::priority_queue<std::pair<int, char>, std::vector<std::pair<int, char>>, cmp>());
    for(const auto &e : g.eList)
    {
        pqVector.at(e.src).push(std::pair<int, char>(e.dst, MARK_UNVISITED));
        pqVector.at(e.dst).push(std::pair<int, char>(e.src, MARK_UNVISITED));
    }

    //For every vertex (for example i), pull sorted vState pairs from pq and push them into g.verticesValue.at(i).vStateList
    //The order of verticesValue.vStateList in graph can be ensured
    for(int i = 0; i < g.vCount; i++)
    {
        while(!pqVector.at(i).empty())
        {
            g.verticesValue.at(i).vStateList.emplace_back(pqVector.at(i).top());
            g.verticesValue.at(i).relatedVCount++;
            pqVector.at(i).pop();
        }
    }

    //initV init
    int initV = initVList.at(0);

    g.vList.at(initV).isActive = true;

    auto &vV = g.verticesValue.at(initV);
    vV.state = STATE_DISCOVERED;
    this->search(initV, this->numOfInitV, &initVList[0], &g.vList[0], &g.verticesValue[0], avCount);
    vV.opbit |= OP_BROADCAST;
}

template <typename VertexValueType, typename MessageValueType>
void DDFS<VertexValueType, MessageValueType>::Deploy(int vCount, int eCount, int numOfInitV)
{

}

template <typename VertexValueType, typename MessageValueType>
void DDFS<VertexValueType, MessageValueType>::Free()
{

}

template<typename VertexValueType, typename MessageValueType>
std::vector<Graph<VertexValueType>>
DDFS<VertexValueType, MessageValueType>::DivideGraphByEdge(const Graph<VertexValueType> &g, int partitionCount)
{
    auto res = std::vector<Graph<VertexValueType>>();

    //Divide edges into multiple subgraphs
    auto eG = std::vector<std::vector<Edge>>();
    for(int i = 0; i < partitionCount; i++) eG.emplace_back(std::vector<Edge>());
    for(int i = 0; i < partitionCount; i++)
    {
        for(int j = (i * g.eCount) / partitionCount; j < ((i + 1) * g.eCount) / partitionCount; j++)
            eG.at(i).emplace_back(g.eList.at(j));
    }

    //Init subGs parameters
    auto templateBlankVV = std::vector<VertexValueType>(g.vCount, VertexValueType());
    for(int i = 0; i < g.vCount; i++)
    {
        templateBlankVV.at(i).state = g.verticesValue.at(i).state;
        templateBlankVV.at(i).opbit = g.verticesValue.at(i).opbit;
        templateBlankVV.at(i).vNextMSGTo = g.verticesValue.at(i).vNextMSGTo;
        templateBlankVV.at(i).startTime = g.verticesValue.at(i).startTime;
        templateBlankVV.at(i).endTime = g.verticesValue.at(i).endTime;
    }

    for(int i = 0; i < partitionCount; i++)
        res.emplace_back(Graph<VertexValueType>(g.vList, eG.at(i), templateBlankVV));

    for(auto &subG : res)
    {
        //Scan edges in each subgraph and collect info
        /*
         * for edge (a, b):
         *     add pair (b, MARK_UNVISITED) as (vid, mark) into a's vState priority queue ordered by vid
         *     add pair (a, MARK_UNVISITED) as (vid, mark) into b's vState priority queue ordered by vid
        */
        auto pqVector = std::vector<std::priority_queue<std::pair<int, char>, std::vector<std::pair<int, char>>, cmp>>(g.vCount, std::priority_queue<std::pair<int, char>, std::vector<std::pair<int, char>>, cmp>());
        for(const auto &e : subG.eList)
        {
            pqVector.at(e.src).push(std::pair<int, char>(e.dst, MARK_UNVISITED));
            pqVector.at(e.dst).push(std::pair<int, char>(e.src, MARK_UNVISITED));
        }

        //For every vertex (for example i), pull sorted vState pairs from pq and push them into g.verticesValue.at(i).vStateList
        //The order of verticesValue.vStateList in subgraph can be ensured
        for(int i = 0; i < subG.vCount; i++)
        {
            while(!pqVector.at(i).empty())
            {
                subG.verticesValue.at(i).vStateList.emplace_back(pqVector.at(i).top());
                subG.verticesValue.at(i).relatedVCount++;
                pqVector.at(i).pop();
            }
        }
    }

    //Copy vState from global graph into corresponding subgraph.verticesValue.vStateList
    int subGCount = partitionCount;
    int *subGIndex = new int [subGCount];

    for(int i = 0; i < g.vCount; i++)
    {
        for(int j = 0; j < subGCount; j++) subGIndex[j] = 0;
        for(int j = 0; j < g.verticesValue.at(i).relatedVCount; j++)
        {
            const auto &vV = g.verticesValue.at(i).vStateList.at(j);
            for(int k = 0; k < subGCount; k++)
            {
                if(res.at(k).verticesValue.at(i).vStateList.at(subGIndex[k]).first == vV.first)
                {
                    res.at(k).verticesValue.at(i).vStateList.at(subGIndex[k]).second = vV.second;
                    subGIndex[k]++;
                    break;
                }
            }
        }
    }

    return res;
}

template <typename VertexValueType, typename MessageValueType>
int DDFS<VertexValueType, MessageValueType>::search(int vid, int numOfInitV, const int *initVSet, Vertex *vSet, VertexValueType *vValues, int &avCount)
{
    bool chk = false;
    for(auto &vState : vValues[vid].vStateList)
    {
        if(!(vState.second == MARK_VISITED))
        {
            chk = true;
            vState.second = MARK_SON;
            vValues[vid].opbit |= OP_MSG_FROM_SEARCH;
            vValues[vid].opbit |= OP_MSG_DOWNWARD;
            //Vertex which will send msg will be activated
            if(!vSet[vid].isActive)
                avCount++;
            vSet[vid].isActive = true;
            return vState.first;
        }
    }

    if(!chk)
    {
        if(vid == initVSet[0]) return -1;
        else
        {
            //There should be some approach more efficient
            for(auto &vState : vValues[vid].vStateList)
            {
                if(vState.second == MARK_PARENT)
                {
                    //Vertex which will send msg will be activated
                    if(!vSet[vid].isActive)
                        avCount++;
                    vSet[vid].isActive = true;
                    vValues[vid].opbit |= OP_MSG_FROM_SEARCH;
                    return vState.first;
                }
            }

        }
    }

    return -1;
}
