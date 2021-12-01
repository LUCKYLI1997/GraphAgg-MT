//
// Created by Thoh Testarossa on 2019-05-25.
//
// BellmanFordGPU can not use some data types BellmanFord can use and we should assume the possible template type is only
// "double" since the reduce function in CUDA is type-sensitive
//

#include "BellmanFordGPU.cpp"

template class BellmanFordGPU<double, double>;
