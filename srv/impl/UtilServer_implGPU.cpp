//
// Created by Thoh Testarossa on 2019-04-05.
//

#include "../UtilServer.cpp"

#include "../../algo/BellmanFord/BellmanFordGPU.h"

template class UtilServer<BellmanFordGPU<double, double>, double, double>;
