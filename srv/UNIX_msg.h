//
// Created by Thoh Testarossa on 2019-04-04.
//

#pragma once

#ifndef GRAPH_ALGO_UNIX_MSG_H
#define GRAPH_ALGO_UNIX_MSG_H

#include "sys/msg.h"

typedef struct Umsg
{
    long type;
    char mtext[256];
}Umsg;

class UNIX_msg
{
public:
    UNIX_msg();

    int create(key_t key, int auth);
    int fetch(key_t key);
    int control(int cmd);
    int send(const char *msgp, long type, size_t msgsz);
    ssize_t recv(char *msgp, long type, size_t msgsz);

    //Test
    const static int testMSGKey = 0xe89d03f4;
    //Test end

private:
    int msqid;
    struct msqid_ds *buf;
};

#endif //GRAPH_ALGO_UNIX_MSG_H
