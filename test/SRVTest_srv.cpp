//
// Created by Thoh Testarossa on 2019-04-04.
//

#include "../srv/UNIX_shm.h"
#include "../srv/UNIX_msg.h"
#include "../srv/UtilServer.h"
#include "../srv/UtilServer.cpp"
#include "../srv/UtilClient.h"

#include <iostream>
#include <string>

int main()
{
    UNIX_shm testSHM = UNIX_shm();
    UNIX_msg testMSG = UNIX_msg();
    char msg[256];
    std::string cmd = std::string("");
    if(testSHM.create(UNIX_shm::testSHMKey, 16 * sizeof(char), 0666) != -1 &&
       testMSG.create(UNIX_msg::testMSGKey, 0666) != -1)
    {
        testSHM.attach(0666);
        for(int i = 0; i < 15; i++) testSHM.shmaddr[i] = '0';
        testSHM.shmaddr[15] = '\0';
        std::cout << testSHM.shmaddr << std::endl;
        while(testMSG.recv(msg, CLI_MSG_TYPE, 256) != -1)
        {
            cmd = std::string(msg);
            if(std::string("z") == cmd)
            {
                testMSG.send("end", SRV_MSG_TYPE, 256);
                break;
            }
            else
            {
                std::cout << testSHM.shmaddr << std::endl;
                testMSG.send("continue", SRV_MSG_TYPE, 256);
            }
        }
        std::cout << "end" << std::endl;
        testSHM.control(IPC_RMID);
        testMSG.control(IPC_RMID);
    }
    else
        std::cout << "shm or msq is occupied!" << std::endl;

    return 0;
}