//
// Created by Thoh Testarossa on 2019-04-04.
//

#include "UNIX_shm.h"

UNIX_shm::UNIX_shm()
{
    this->shmid = -1;
    this->shmaddr = nullptr;
    this->buf = new struct shmid_ds;
}

int UNIX_shm::create(key_t key, size_t size, int auth)
{
    this->shmid = shmget(key, size, (IPC_CREAT | IPC_EXCL | auth));

    return this->shmid;
}

int UNIX_shm::fetch(key_t key)
{
    this->shmid = shmget(key, 0, 0);

    return this->shmid;
}

void UNIX_shm::attach(int auth)
{
    this->shmaddr = (char *)shmat(this->shmid, nullptr, 0);
}

int UNIX_shm::detach()
{
    return shmdt(this->shmaddr);
}

int UNIX_shm::control(int cmd)
{
    return shmctl(this->shmid, cmd, this->buf);
}

