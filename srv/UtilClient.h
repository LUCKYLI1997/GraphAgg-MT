//
// Created by Thoh Testarossa on 2019-04-05.
//

#pragma once

#ifndef GRAPH_ALGO_UTILCLIENT_H
#define GRAPH_ALGO_UTILCLIENT_H

#include "../core/GraphUtil.h"
#include "../srv/UNIX_shm.h"
#include "../srv/UNIX_msg.h"
#include "../include/UNIX_marco.h"

template <typename VertexValueType, typename MessageValueType>
class UtilClient
{
public:
    UtilClient(int vCount, int eCount, int numOfInitV, int nodeNo = 0);
    ~UtilClient() = default;

    int connect();
    int transfer(VertexValueType *vValues, Vertex *vSet, Edge *eSet, int *initVSet, bool *filteredV, int filteredVCount);
    int update(VertexValueType *vValues, Vertex *vSet);
    void request();
    void disconnect();
    void shutdown();

    int nodeNo;

    int vCount;
    int eCount;
    int numOfInitV;

    int *initVSet;
    bool *filteredV;
    int *filteredVCount;
    VertexValueType *vValues;
    MessageValueType *mValues;
    Vertex *vSet;
    Edge *eSet;

//test
public:
//private:
    UNIX_shm initVSet_shm;
    UNIX_shm filteredV_shm;
    UNIX_shm filteredVCount_shm;
    UNIX_shm vValues_shm;
    UNIX_shm mValues_shm;
    UNIX_shm vSet_shm;
    UNIX_shm eSet_shm;

    UNIX_msg server_msq;
    UNIX_msg client_msq;
};

#endif //GRAPH_ALGO_UTILCLIENT_H
