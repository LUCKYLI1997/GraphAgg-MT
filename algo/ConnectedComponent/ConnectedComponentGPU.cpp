//
// Created by Thoh Testarossa on 2019-08-12.
//

#include "ConnectedComponentGPU.h"
#include "kernel_src/ConnectedComponentGPU_kernel.h"

#include <iostream>
#include <algorithm>

template <typename VertexValueType, typename MessageValueType>
ConnectedComponentGPU<VertexValueType, MessageValueType>::ConnectedComponentGPU()
{

}

template <typename VertexValueType, typename MessageValueType>
void ConnectedComponentGPU<VertexValueType, MessageValueType>::Init(int vCount, int eCount, int numOfInitV)
{
    ConnectedComponent<VertexValueType, MessageValueType>::Init(vCount, eCount, numOfInitV);

    this->vertexLimit = VERTEXSCALEINGPU;
    this->mPerMSGSet = MSGSCALEINGPU;
    this->ePerEdgeSet = EDGESCALEINGPU;
}

template <typename VertexValueType, typename MessageValueType>
void ConnectedComponentGPU<VertexValueType, MessageValueType>::GraphInit(Graph<VertexValueType> &g, std::set<int> &activeVertices,
                                                       const std::vector<int> &initVList)
{
    ConnectedComponent<VertexValueType, MessageValueType>::GraphInit(g, activeVertices, initVList);
}

template <typename VertexValueType, typename MessageValueType>
void ConnectedComponentGPU<VertexValueType, MessageValueType>::Deploy(int vCount, int eCount, int numOfInitV)
{
    ConnectedComponent<VertexValueType, MessageValueType>::Deploy(vCount, eCount, numOfInitV);

    cudaError_t err = cudaSuccess;

    this->vValueSet = new VertexValueType [vCount];
    err = cudaMalloc((void **)&this->d_vValueSet, vertexLimit * sizeof(int));

    this->mValueTable = new VertexValueType [vertexLimit];
    err = cudaMalloc((void **)&this->d_mValueTable, vertexLimit * sizeof(int));

    err = cudaMalloc((void **)&this->d_vSet, vertexLimit * sizeof(Vertex));
    err = cudaMalloc((void **)&this->d_eGSet, ePerEdgeSet * sizeof(Edge));

    int mSize = std::max(this->numOfInitV * ePerEdgeSet, mPerMSGSet);

    this->mInitVIndexSet = new int [mSize];
    err = cudaMalloc((void **)&this->d_mInitVIndexSet, mSize * sizeof(int));
    this->mDstSet = new int [mSize];
    err = cudaMalloc((void **)&this->d_mDstSet, mSize * sizeof(int));
    this->mValueSet = new VertexValueType [mSize];
    err = cudaMalloc((void **)&this->d_mValueSet, mSize * sizeof(int));
}

template <typename VertexValueType, typename MessageValueType>
void ConnectedComponentGPU<VertexValueType, MessageValueType>::Free()
{
    ConnectedComponent<VertexValueType, MessageValueType>::Free();

    free(this->vValueSet);
    cudaFree(this->d_vValueSet);

    free(this->mValueTable);
    cudaFree(this->d_mValueTable);

    cudaFree(this->d_vSet);
    cudaFree(this->d_eGSet);

    free(this->mInitVIndexSet);
    cudaFree(this->d_mInitVIndexSet);
    free(this->mDstSet);
    cudaFree(this->d_mDstSet);
    free(this->mValueSet);
    cudaFree(this->d_mValueSet);
}

template<typename VertexValueType, typename MessageValueType>
int ConnectedComponentGPU<VertexValueType, MessageValueType>::MSGApply_array(int vCount, int eCount, Vertex *vSet, int numOfInitV,
                                                            const int *initVSet, VertexValueType *vValues,
                                                            MessageValueType *mValues)
{
    //Availability check
    if(vCount <= 0) return 0;

    //CUDA init
    cudaError_t err = cudaSuccess;

    bool needReflect = vCount > this->vertexLimit;

    //AVCheck
    for (int i = 0; i < vCount; i++) vSet[i].isActive = false;

    if(!needReflect)
    {
        //vSet copy
        err = cudaMemcpy(this->d_vSet, vSet, vCount * sizeof(Vertex), cudaMemcpyHostToDevice);

        //vValues copy
        err = cudaMemcpy(this->d_vValueSet, (int *)vValues, vCount * sizeof(int), cudaMemcpyHostToDevice);
    }

    //Apply msgs to v
    int mGCount = 0;
    auto mGSet = MessageSet<MessageValueType>();

    auto r_mGSet = MessageSet<MessageValueType>();
    auto r_vSet = std::vector<Vertex>();
    auto r_vValueSet = std::vector<VertexValueType>();

    for(int i = 0; i < vCount; i++)
    {
        if(mValues[i] != INVALID_MASSAGE) //Adding msgs to batchs
        {
            mGSet.insertMsg(Message<MessageValueType>(INVALID_INITV_INDEX, i, mValues[i]));
            mGCount++;
        }
        if(mGCount == this->mPerMSGSet || i == vCount - 1) //A batch of msgs will be transferred into GPU. Don't forget last batch!
        {
            auto reflectIndex = std::vector<int>();
            auto reversedIndex = std::vector<int>();

            //Reflection for message & vertex & vValues
            if(needReflect)
            {
                //MSG reflection
                r_mGSet = this->reflectM(mGSet, vCount, reflectIndex, reversedIndex);

                for(int j = 0; j < r_mGSet.mSet.size(); j++)
                {
                    this->mInitVIndexSet[j] = vSet[r_mGSet.mSet.at(j).src].initVIndex;
                    this->mDstSet[j] = r_mGSet.mSet.at(j).dst;
                    this->mValueSet[j] = r_mGSet.mSet.at(j).value;
                }

                //v reflection
                r_vSet.clear();
                for(int j = 0; j < reflectIndex.size(); j++)
                    r_vSet.emplace_back(j, false, vSet[reflectIndex.at(j)].initVIndex);

                //vValue reflection
                r_vValueSet.clear();
                r_vValueSet.reserve(mPerMSGSet);
                r_vValueSet.assign(mPerMSGSet, INT32_MAX >> 1);
                for(int j = 0; j < reflectIndex.size(); j++)
                    r_vValueSet.at(j) = vValues[reflectIndex[j]];

                //vSet & vValueSet Init
                err = cudaMemcpy(this->d_vSet, &r_vSet[0], reflectIndex.size() * sizeof(Vertex), cudaMemcpyHostToDevice);

                err = cudaMemcpy(this->d_vValueSet, (int *)&r_vValueSet[0], reflectIndex.size() * sizeof(int), cudaMemcpyHostToDevice);
            }
            else
            {
                //Use original msg
                for(int j = 0; j < mGSet.mSet.size(); j++)
                {
                    this->mInitVIndexSet[j] = vSet[mGSet.mSet.at(j).src].initVIndex;
                    this->mDstSet[j] = mGSet.mSet.at(j).dst;
                    this->mValueSet[j] = mGSet.mSet.at(j).value;
                }
            }

            //MSG memory copy
            err = cudaMemcpy(this->d_mInitVIndexSet, this->mInitVIndexSet, mGCount * sizeof(int), cudaMemcpyHostToDevice);
            err = cudaMemcpy(this->d_mDstSet, this->mDstSet, mGCount * sizeof(int), cudaMemcpyHostToDevice);
            err = cudaMemcpy(this->d_mValueSet, (int *)this->mValueSet, mGCount * sizeof(int), cudaMemcpyHostToDevice);

            //Kernel Execution
            for(int j = 0; j < mGCount; j += NUMOFGPUCORE)
            {
                int msgNumUsedForExec = (mGCount - j > NUMOFGPUCORE) ? NUMOFGPUCORE : (mGCount - j);

                err = MSGApply_kernel_exec(this->d_vSet, numOfInitV, nullptr, this->d_vValueSet, msgNumUsedForExec,
                                           &this->d_mDstSet[j], &this->d_mInitVIndexSet[j], &this->d_mValueSet[j]);
            }

            //Deflection
            if(needReflect)
            {
                err = cudaMemcpy(&r_vSet[0], this->d_vSet, reflectIndex.size() * sizeof(Vertex), cudaMemcpyDeviceToHost);
                err = cudaMemcpy((int *)&r_vValueSet[0], this->d_vValueSet, reflectIndex.size() * sizeof(int), cudaMemcpyDeviceToHost);

                for(int j = 0; j < reflectIndex.size(); j++)
                {
                    vSet[reflectIndex[j]] = r_vSet[j];
                    //Don't forget to deflect vertexID in Vertex obj!!
                    vSet[reflectIndex[j]].vertexID = reflectIndex[j];

                    vValues[reflectIndex[j]] = r_vValueSet[j];
                }
            }

            mGSet.mSet.clear();
            mGCount = 0;
        }
    }

    //Re-package the data

    //Memory copy back
    if(!needReflect)
    {
        err = cudaMemcpy(vSet, this->d_vSet, vCount * sizeof(Vertex), cudaMemcpyDeviceToHost);
        err = cudaMemcpy((int *)vValues, this->d_vValueSet, vCount * sizeof(int), cudaMemcpyDeviceToHost);
    }

    //avCount calculation
    int avCount = 0;
    for(int i = 0; i < vCount; i++)
    {
        if(vSet[i].isActive)
            avCount++;
    }

    return avCount;
}

template<typename VertexValueType, typename MessageValueType>
int
ConnectedComponentGPU<VertexValueType, MessageValueType>::MSGGenMerge_array(int vCount, int eCount, const Vertex *vSet, const Edge *eSet,
                                                          int numOfInitV, const int *initVSet,
                                                          const VertexValueType *vValues, MessageValueType *mValues)
{
    //Generate merged msgs directly

    //Availability check
    if(vCount <= 0) return 0;

    //Invalid message init
    for(int i = 0; i < vCount; i++) mValues[i] = (MessageValueType)INVALID_MASSAGE;

    //Memory allocation
    cudaError_t err = cudaSuccess;

    //Graph scale check
    bool needReflect = vCount > this->vertexLimit;

    if(!needReflect)
    {
        //vSet init
        err = cudaMemcpy(this->d_vSet, vSet, vCount * sizeof(Vertex), cudaMemcpyHostToDevice);

        //vValueSet init
        err = cudaMemcpy(this->d_vValueSet, (int *)vValues, vCount * sizeof(int), cudaMemcpyHostToDevice);

        //mValueTable init
        err = cudaMemcpy(this->d_mValueTable, (int *)mValues, vCount * sizeof(int), cudaMemcpyHostToDevice);
    }

    //Init for possible reflection
    //Maybe can use lambda style?
    bool *tmp_AVCheckList = new bool [vCount];
    auto tmp_o_g = Graph<VertexValueType>(0);
    if(needReflect)
    {
        for(int i = 0; i < vCount; i++) tmp_AVCheckList[i] = vSet[i].isActive;
        tmp_o_g = Graph<VertexValueType>(vCount, 0, nullptr, nullptr, nullptr, tmp_AVCheckList);
        tmp_o_g.verticesValue.reserve(vCount);
        tmp_o_g.verticesValue.insert(tmp_o_g.verticesValue.begin(), vValues, vValues + vCount);
    }
    //This checkpoint is to used to prevent from mistaking mValues gathering in deflection
    bool *isDst = new bool [vCount];
    for(int i = 0; i < vCount; i++) isDst[i] = false;

    //e batch processing
    int eGCount = 0;

    std::vector<Edge> eGSet = std::vector<Edge>();
    eGSet.reserve(this->ePerEdgeSet);

    for(int i = 0; i < eCount; i++)
    {
        if(vSet[eSet[i].src].isActive) //Add es to batches
        {
            eGSet.emplace_back(eSet[i]);
            eGCount++;
            //Only dst receives message
            isDst[eSet[i].dst] = true;
        }
        if(eGCount == this->ePerEdgeSet || i == eCount - 1) //A batch of es will be transferred into GPU. Don't forget last batch!
        {
            auto reflectIndex = std::vector<int>();
            auto reversedIndex = std::vector<int>();

            auto r_g = Graph<VertexValueType>(0);

            //Reflection
            if(needReflect)
            {
                r_g = this->reflectG(tmp_o_g, eGSet, reflectIndex, reversedIndex);

                //vSet init
                err = cudaMemcpy(this->d_vSet, &r_g.vList[0], r_g.vCount * sizeof(Vertex), cudaMemcpyHostToDevice);

                //vValueSet init
                err = cudaMemcpy(this->d_vValueSet, (int *)&r_g.verticesValue[0], r_g.vCount * sizeof(int), cudaMemcpyHostToDevice);

                //mValueTable init
                err = cudaMemcpy(this->d_mValueTable, (int *)mValues, r_g.vCount * sizeof(int), cudaMemcpyHostToDevice);

                err = cudaMemcpy(this->d_eGSet, &r_g.eList[0], eGCount * sizeof(Edge), cudaMemcpyHostToDevice);
            }
            else
                err = cudaMemcpy(this->d_eGSet, &eGSet[0], eGCount * sizeof(Edge), cudaMemcpyHostToDevice);

            //Kernel Execution (no matter whether g is reflected or not)
            for(int j = 0; j < eGCount; j += NUMOFGPUCORE)
            {
                int edgeNumUsedForExec = (eGCount - j > NUMOFGPUCORE) ? NUMOFGPUCORE : (eGCount - j);

                err = MSGGenMerge_kernel_exec(this->d_mValueTable, this->d_vSet, numOfInitV,
                                              nullptr, this->d_vValueSet, edgeNumUsedForExec, &this->d_eGSet[j]);
            }

            //Deflection
            if(needReflect)
            {
                //Re-package the data
                //Memory copy back
                err = cudaMemcpy((int *)this->mValueTable, this->d_mValueTable, r_g.vCount * sizeof(int), cudaMemcpyDeviceToHost);

                //Valid message transformed back to original double form (deflection)
                for (int j = 0; j < r_g.vCount; j++)
                {
                    int o_dst = reflectIndex[j];
                    //If the v the current msg point to is not a dst, it should not be copied back because the current msg value is not correct)
                    if(isDst[o_dst])
                    {
                        if(mValues[o_dst] > this->mValueTable[j])
                            mValues[o_dst] > this->mValueTable[j];
                    }
                }
            }
            else;

            //Checkpoint reset
            eGCount = 0;
            eGSet.clear();
            for(int j = 0; j < vCount; j++) isDst[j] = false;
        }
    }

    if(!needReflect)
    {
        //Re-package the data
        //Memory copy back
        err = cudaMemcpy((int *)this->mValueTable, this->d_mValueTable, vCount * sizeof(int), cudaMemcpyDeviceToHost);

        //Transform back to original double form
        for (int i = 0; i < vCount; i++)
            mValues[i] = this->mValueTable[i];
    }

    return vCount;
}

