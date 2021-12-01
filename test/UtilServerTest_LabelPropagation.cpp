//
// Created by cave-g-f on 2019-05-29.
//

#include "../algo/LabelPropagation/LabelPropagation.h"
#include "../core/GraphUtil.h"
#include "../srv/UtilServer.h"
#include "../srv/UNIX_shm.h"
#include "../srv/UNIX_msg.h"

#include <iostream>

int main(int argc, char *argv[])
{
    if(argc != 4 && argc != 5)
    {
        std::cout << "Usage:" << std::endl << "./UtilServerTest_LabelPropagation vCount eCount numOfInitV [nodeNo]" << std::endl;
        return 1;
    }

    int vCount = atoi(argv[1]);
    int eCount = atoi(argv[2]);
    int numOfInitV = atoi(argv[3]);
    int nodeNo = (argc == 4) ? 0 : atoi(argv[4]);

    auto testUtilServer = UtilServer<LabelPropagation<std::pair<int, int>, std::pair<int, int>>, std::pair<int, int>, std::pair<int, int>>(vCount, eCount, numOfInitV, nodeNo);
    if(!testUtilServer.isLegal)
    {
        std::cout << "mem allocation failed or parameters are illegal" << std::endl;
        return 2;
    }

    testUtilServer.run();
}
