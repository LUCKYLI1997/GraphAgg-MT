//
// Created by Thoh Testarossa on 2019-04-04.
//

#include "../srv/UNIX_shm.h"
#include "../srv/UNIX_msg.h"
#include "../srv/UtilServer.h"
#include "../srv/UtilClient.h"

#include <iostream>
#include <string>

int main()
{
    UNIX_shm testSHM = UNIX_shm();
    UNIX_msg testMSG = UNIX_msg();
    char msg[256];

    std::string cmd = std::string("");
    if(testSHM.fetch(UNIX_shm::testSHMKey) != -1 &&
       testMSG.fetch(UNIX_msg::testMSGKey) != -1)
    {
        testSHM.attach(0666);

        int ret = -1;
        std::string ret_s = std::string("");

        do{
            std::cin >> cmd;

            if(std::string("z") == cmd)
                testMSG.send("z", CLI_MSG_TYPE, 256);
            else
            {
                for(int i = 0; i < 16 && i < cmd.length(); i++)
                {
                    testSHM.shmaddr[i] = cmd[i];
                    testSHM.shmaddr[std::min(i + 1, 15)] = '\0';
                }
                testMSG.send("k", CLI_MSG_TYPE, 256);
            }

            ret = testMSG.recv(msg, SRV_MSG_TYPE, 256);
            if(ret != -1)
            {
                ret_s = std::string(msg);
                std::cout << ret_s << std::endl;
            }

        }while(ret != -1 && ret_s != std::string("end"));
        testSHM.detach();
    }
    else
        std::cout << "Server must be launched first!" << std::endl;

    return 0;
}