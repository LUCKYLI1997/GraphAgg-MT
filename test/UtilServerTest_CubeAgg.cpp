#include "../algo/CubeAgg/CubeAgg.h"
#include "../core/GraphUtil.h"
#include "../srv/UtilServer.h"
#include "../srv/UNIX_shm.h"
#include "../srv/UNIX_msg.h"

#include <iostream>

int main(int argc, char* argv[])
{
    if (argc != 6)
    {
        std::cout << "Usage:" << std::endl << "./UtilServerTest_CubeAgg vCount eCount dCount partitionNo threadCount" << std::endl;
        return 1;
    }

    int vCount = atoi(argv[1]);
    int eCount = atoi(argv[2]);
    int dCount = atoi(argv[3]);
    int partitionNo = atoi(argv[4]);
    int threadCount = atoi(argv[5]);

    auto testUtilServer = UtilServer<CubeAgg<CubeAggVertexValue, int>, CubeAggVertexValue, int>(vCount, eCount, 1000, partitionNo);   //actual value: 13 = 3 + maxdCount(10)
    if (!testUtilServer.isLegal)
    {
        std::cout << "mem allocation failed or parameters are illegal" << std::endl;
        return 2;
    }

    testUtilServer.executor.dCount = dCount;
    testUtilServer.executor.currentThread = threadCount;
    testUtilServer.run();
}