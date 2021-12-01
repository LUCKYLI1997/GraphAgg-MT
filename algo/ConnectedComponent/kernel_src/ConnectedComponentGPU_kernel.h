//
// Created by Thoh Testarossa on 2019-03-13.
//

#pragma once

#ifndef GRAPH_ALGO_CONNECTEDCOMPONENTGPU_KERNEL_H
#define GRAPH_ALGO_CONNECTEDCOMPONENTGPU_KERNEL_H

#include "../../../include/GPUconfig.h"
#include "../ConnectedComponentGPU.h"

#include <cuda_runtime.h>

__global__ void MSGApply_kernel(Vertex *vSet, int numOfInitV, int *initVSet, int *vValues, 
                                int numOfMsg, int *mDstSet, int *mInitVIndexSet, int *mValueSet);

cudaError_t MSGApply_kernel_exec(Vertex *vSet, int numOfInitV, int *initVSet, int *vValues,
	                             int numOfMsg, int *mDstSet, int *mInitVIndexSet, int *mValueSet);

__global__ void MSGGenMerge_kernel(int *mValues,
	                               Vertex *vSet, int numOfInitV, int *initVSet, int *vValues,
	                               int numOfEdge, Edge *eSet);

cudaError_t MSGGenMerge_kernel_exec(int *mValues,
	                               Vertex *vSet, int numOfInitV, int *initVSet, int *vValues,
	                               int numOfEdge, Edge *eSet);
#endif //GRAPH_ALGO_BELLMANFORDGPU_KERNEL_H
