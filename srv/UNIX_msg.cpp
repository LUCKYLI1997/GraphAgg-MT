//
// Created by Thoh Testarossa on 2019-04-04.
//

#include "UNIX_msg.h"
#include <cstring>

UNIX_msg::UNIX_msg()
{
    this->msqid = -1;
    this->buf = new msqid_ds;
}

int UNIX_msg::create(key_t key, int auth)
{
    this->msqid = msgget(key, (IPC_CREAT | IPC_EXCL | auth));

    return this->msqid;
}

int UNIX_msg::fetch(key_t key)
{
    this->msqid = msgget(key, 0);

    return this->msqid;
}

int UNIX_msg::control(int cmd)
{
    return msgctl(this->msqid, cmd, this->buf);
}

int UNIX_msg::send(const char *msgp, long type, size_t msgsz)
{
    Umsg u_msg;
    u_msg.type = type;
    strcpy(u_msg.mtext, msgp);
    return msgsnd(this->msqid, (void *)&u_msg, msgsz, 0);
}

ssize_t UNIX_msg::recv(char *msgp, long type, size_t msgsz)
{
    Umsg u_msg;
    int ret = msgrcv(this->msqid, (void *)&u_msg, msgsz, type, 0);
    if(ret != -1) strcpy(msgp, u_msg.mtext);
    return ret;
}

