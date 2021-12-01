//
// Created by Thoh Testarossa on 2019-04-05.
//

#include "../UtilServer.cpp"

#include "../../algo/BellmanFord/BellmanFord.h"
#include "../../algo/LabelPropagation/LabelPropagation.h"
#include "../../algo/CubeAgg/CubeAgg.h"

template class UtilServer<BellmanFord<double, double>, double, double>;
template class UtilServer<LabelPropagation<std::pair<int, int>, std::pair<int, int>>, std::pair<int, int>, std::pair<int, int>>;
template class UtilServer<CubeAgg<CubeAggVertexValue, int>, CubeAggVertexValue, int>;