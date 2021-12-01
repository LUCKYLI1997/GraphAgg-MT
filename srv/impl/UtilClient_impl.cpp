//
// Created by liqi on 19-5-27.
//

#include "../UtilClient.cpp"
#include "../../algo/CubeAgg/CubeAgg.h"

template class UtilClient<double, double>;
template class UtilClient<std::pair<int, int>, std::pair<int, int>>;
template class UtilClient<CubeAggVertexValue, int>;