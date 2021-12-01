#include "ConnectedComponentGPU_kernel.h"

__global__ void MSGApply_kernel(Vertex *vSet, int numOfInitV, int *initVSet, int *vValues,
	int numOfMsg, int *mDstSet, int *mInitVIndexSet, int *mValueSet)
{
	int tid = threadIdx.x;

	if(tid < numOfMsg)
	{
		int vID = mDstSet[tid];

		if(vValues[vID] > mValueSet[tid])
		{
			vValues[vID] = mValueSet[tid];
			vSet[vID].isActive = true;
		}

		else;
	}
}

cudaError_t MSGApply_kernel_exec(Vertex *vSet, int numOfInitV, int *initVSet, int *vValues,
	int numOfMsg, int *mDstSet, int *mInitVIndexSet, int *mValueSet)
{
	cudaError_t err = cudaSuccess;
	
	MSGApply_kernel<<<1, NUMOFGPUCORE>>>(vSet, numOfInitV, initVSet, vValues, numOfMsg, mDstSet, mInitVIndexSet, mValueSet);
    err = cudaGetLastError();

	cudaDeviceSynchronize();
	
	return err;
}

__global__ void MSGGenMerge_kernel(int *mValues,
	Vertex *vSet, int numOfInitV, int *initVSet, int *vValues,
	int numOfEdge, Edge *eSet)
{
	int tid = threadIdx.x;

	if(tid < numOfEdge)
	{
		int vID = -1;
		if(vSet[eSet[tid].src].isActive) vID = eSet[tid].dst;

		if(vID != -1)
		{
			for(int i = 0; i < numOfInitV; i++)
				atomicMin(&mValues[vID], vValues[eSet[tid].src] + eSet[tid].weight);
		}
		else;
	}
}

cudaError_t MSGGenMerge_kernel_exec(int *mValues,
	Vertex *vSet, int numOfInitV, int *initVSet, int *vValues,
	int numOfEdge, Edge *eSet)
{
	cudaError_t err = cudaSuccess;

	MSGGenMerge_kernel<<<1, NUMOFGPUCORE>>>(mValues, vSet, numOfInitV, initVSet, vValues, numOfEdge, eSet);
	err = cudaGetLastError();

	cudaDeviceSynchronize();
	
	return err;
}