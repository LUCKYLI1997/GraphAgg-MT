//
// Created by Thoh Testarossa on 2019-03-08.
//

#include <iostream>
#include <fstream>

#include <random>

#define MAXEDGEWEIGHT 1000

int main(int argc, char **argv)
{
    if(argc != 3) {std::cout << "Usage:" << std::endl << "./RandGraphGen vCount eCount" << std::endl; return 1; }

    int vCount = atoi(argv[1]), eCount = atoi(argv[2]);

    std::random_device r1;
    std::uniform_int_distribution<int> uniform_dist_1(0, vCount - 1);
    std::default_random_engine e1(r1());

    std::random_device r2;
    std::uniform_int_distribution<int> uniform_dist_2(0, MAXEDGEWEIGHT);
    std::default_random_engine e2(r2());

    std::ofstream Gout("testGraph.txt");

    Gout << vCount << " " << eCount << std::endl;
    for(int i = 0; i < eCount; i++)
    {
        int src = uniform_dist_1(e1), dst;
        double weight = uniform_dist_2(e2);
        do
        {
            dst = uniform_dist_1(e1);
        }while(dst == src);
        Gout << src << " " << dst << " " << weight << std::endl;
    }

    Gout.close();

    return 0;
}