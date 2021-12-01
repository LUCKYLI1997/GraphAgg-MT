//
// Created by Thoh Testarossa on 2019-04-04.
//

#pragma once

#ifndef GRAPH_ALGO_UNIX_SHM_H
#define GRAPH_ALGO_UNIX_SHM_H

#include "sys/shm.h"

class UNIX_shm
{
public:
    UNIX_shm();

    int create(key_t key, size_t size, int auth);
    int fetch(key_t key);
    void attach(int auth);
    int detach();
    int control(int cmd);

    char *shmaddr;

    //Test
    const static int testSHMKey = 0xa3deaf72;
    //Test end

//test
//private:
public:
    int shmid;
    struct shmid_ds *buf;
};

#endif //GRAPH_ALGO_UNIX_SHM_H
