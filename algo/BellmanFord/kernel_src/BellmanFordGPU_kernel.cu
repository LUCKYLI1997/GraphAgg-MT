#include "BellmanFordGPU_kernel.h"

__global__ void MSGApply_kernel(Vertex *vSet, int numOfInitV, int *initVSet, double *vValues,
	int numOfMsg, int *mDstSet, int *mInitVIndexSet, double *mValueSet)
{
	int tid = threadIdx.x;

	if(tid < numOfMsg)
	{
		int vID = mDstSet[tid];
		int vInitVIndex = mInitVIndexSet[tid];

		if(vInitVIndex != -1)
		{
			if(vValues[vID * numOfInitV + vInitVIndex] > mValueSet[tid])
			{
				vValues[vID * numOfInitV + vInitVIndex] = mValueSet[tid];
				vSet[vID].isActive = true;
			}
		}

		else;
	}
}

cudaError_t MSGApply_kernel_exec(Vertex *vSet, int numOfInitV, int *initVSet, double *vValues,
	int numOfMsg, int *mDstSet, int *mInitVIndexSet, double *mValueSet)
{
	cudaError_t err = cudaSuccess;
	
	MSGApply_kernel<<<1, NUMOFGPUCORE>>>(vSet, numOfInitV, initVSet, vValues, numOfMsg, mDstSet, mInitVIndexSet, mValueSet);
    err = cudaGetLastError();

	cudaDeviceSynchronize();
	
	return err;
}

__global__ void MSGGenMerge_kernel(unsigned long long int *mTransformdMergedMSGValueSet,
	Vertex *vSet, int numOfInitV, int *initVSet, double *vValues,
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
				atomicMin(&mTransformdMergedMSGValueSet[vID * numOfInitV + i], __double_as_longlong(vValues[eSet[tid].src * numOfInitV + i] + eSet[tid].weight));
		}
		else;
	}
}

cudaError_t MSGGenMerge_kernel_exec(unsigned long long int *mTransformdMergedMSGValueSet,
	Vertex *vSet, int numOfInitV, int *initVSet, double *vValues,
	int numOfEdge, Edge *eSet)
{
	cudaError_t err = cudaSuccess;

	MSGGenMerge_kernel<<<1, NUMOFGPUCORE>>>(mTransformdMergedMSGValueSet, vSet, numOfInitV, initVSet, vValues, numOfEdge, eSet);
	err = cudaGetLastError();

	cudaDeviceSynchronize();
	
	return err;
}