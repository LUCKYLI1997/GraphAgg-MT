#include<iostream>
#include<fstream>
#include<stdlib.h>
#include<string>
#include<vector>
#include<iomanip>

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::cout << "Usage:" << std::endl << "./xxx nodeCount" << std::endl;
        return 1;
    }

    int nodeCount = atoi(argv[1]);

    std::string name = "ipcrm_" +  std::to_string(nodeCount) + ".sh";
    std::ofstream outputStream(name);

    // delete shm
    std::vector<char> tmp { '1', '2', '3', '6', '7', '9', 'a' };
    for(int nid=0; nid < nodeCount; nid++)
    {
        for(int i = 0; i < tmp.size(); i++)
        {
            outputStream << "ipcrm -M 0x";
            outputStream << std::hex << std::setw(2) << std::setfill('0') << nid;
            outputStream << "0" << tmp[i] << "0000" << std::endl;
        }
    }
    outputStream << std::endl;

    // delete msg
    std::vector<char> tmp2 { '1', '2', '3' };
    for(int nid=0; nid < nodeCount; nid++)
    {
        for(int i = 0; i < tmp2.size(); i++)
        {
            outputStream << "ipcrm -Q 0x";
            outputStream << std::hex << std::setw(2) << std::setfill('0') << nid;
            outputStream << "000" << tmp2[i] << "00" << std::endl;
        }
    }

    return 0;    
}