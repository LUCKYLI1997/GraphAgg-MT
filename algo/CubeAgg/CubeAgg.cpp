#include "CubeAgg.h"

CubeAggVertexValue::CubeAggVertexValue() :measure(0)
{
	for (int i = 0; i < 100; i++)
	{
		this->dimension[i] = 0;
	}
	for (int i = 0; i < 10; i++)
	{
		this->dimension_ptr[i] = &this->dimension[10 * i];
	}
}

CubeAggVertexValue::CubeAggVertexValue(char dimension[100], int measure) :measure(measure)
{
	for (int i = 0; i < 100; i++)
	{
		this->dimension[i] = dimension[i];
	}
	for (int i = 0; i < 10; i++)
	{
		this->dimension_ptr[i] = &this->dimension[10 * i];
	}
}

CubeAggGraph::CubeAggGraph() :vCount(0), eCount(0), vList(std::vector<Vertex>()),
eList(std::vector<Edge>()), verticesValue(std::vector<CubeAggVertexValue>()) {}

AggVertex::AggVertex() : measure(0), TA_count(0)
{
	for (int i = 0; i < 100; i++)
	{
		dimension[i] = 0;
	}
	for (int i = 0; i < 10; i++)
	{
		dimension_ptr[i] = &dimension[10 * i];
	}
}

AggVertex::AggVertex(char dimension[100], int measure, int TA_count) :measure(measure), TA_count(TA_count)
{
	for (int i = 0; i < 100; i++)
	{
		this->dimension[i] = dimension[i];
	}
	for (int i = 0; i < 10; i++)
	{
		dimension_ptr[i] = &dimension[10 * i];
	}
}

AggVertex::AggVertex(const AggVertex& copy) :measure((int)copy.measure), TA_count((int)copy.TA_count)
{
	for (int i = 0; i < 100; i++)
	{
		this->dimension[i] = copy.dimension[i];
	}
	for (int i = 0; i < 10; i++)
	{
		this->dimension_ptr[i] = &this->dimension[10 * i];
	}
}

AggEdge::AggEdge() :src(-1), dst(-1), weight(0), TA_count(0) {}

AggEdge::AggEdge(int src, int dst, int weight, int TA_count) : src(src), dst(dst), weight(weight),
	TA_count(TA_count) {}

AggEdge::AggEdge(const AggEdge& copy) : src(copy.src), dst(copy.dst), weight((int)copy.weight),
	TA_count((int)copy.TA_count) {}


long long GetTime(struct timeval end, struct timeval start)
{
	long long result = 1000000 * ((long long)end.tv_sec - (long long)start.tv_sec) + ((long long)end.tv_usec - (long long)start.tv_usec);
	return result;
}

void EC_compute_thread(int tid, std::vector<Edge>& eList, std::vector<int>& v2v,
	std::vector<long long>& e2hashKey, std::vector<int>& matrixRow, std::vector<long long>& threadTime)
{
	struct timeval start, end;
	gettimeofday(&start, NULL);
	for (int i = 0; i < matrixRow.size(); i++)
	{

		int e_index = matrixRow[i];
		if (e_index != -1)
		{
			int src = eList[e_index].src;
			int dst = eList[e_index].dst;
			int agg_src = v2v[src];
			int agg_dst = v2v[dst];
			long long hashKeyEdge = (long long)agg_src * MAX_NODE_NUMBER + (long long)agg_dst;
			e2hashKey[e_index] = hashKeyEdge;
		}
	}
	gettimeofday(&end, NULL);
	threadTime[tid] = GetTime(end, start);	// record work time
}

void EC_merge_thread(int tid, const CubeAggGraph& sub_g, std::vector<AggEdge>& aggEList,
	std::vector<int>& local_e2e, std::vector<int>& matrixRow, std::vector<long long>& threadTime)
{
	struct timeval start, end;
	gettimeofday(&start, NULL);
	for (int i = 0; i < matrixRow.size(); i++)
	{
		int e_index = matrixRow[i];
		if (e_index != -1)
		{
			int aggE_index = local_e2e[e_index];
			int weight = sub_g.eList[e_index].weight;
			aggEList[aggE_index].weight += weight;
		}
	}
	gettimeofday(&end, NULL);
	threadTime[tid] = GetTime(end, start);	// record work time
}

void VC_compute_thread(int tid, std::vector<CubeAggVertexValue>& vValueList, std::vector<int>& aggDimension,
	std::vector<std::string>& v2hashKey, std::vector<int>& matrixRow, std::vector<long long>& threadTime)
{
	struct timeval start, end;
	gettimeofday(&start, NULL);
	for (int i = 0; i < matrixRow.size(); i++)
	{
		int v_index = matrixRow[i];
		if (v_index != -1)
		{
			v2hashKey[v_index] = CubeAgg<CubeAggVertexValue, int>::GetHashKeyAggVertex(aggDimension, 
				vValueList[v_index].dimension, vValueList[v_index].dimension_ptr);
		}
	}
	gettimeofday(&end, NULL);
	threadTime[tid] = GetTime(end, start);	// record work time
}

void VC_merge_thread(int tid, const CubeAggGraph& sub_g, std::vector<AggVertex>& aggVList,
	std::vector<int>& local_v2v, std::vector<int>& matrixRow, std::vector<long long>& threadTime)
{	
	struct timeval start, end;
	gettimeofday(&start, NULL);
	for (int i = 0; i < matrixRow.size(); i++)
	{
		int v_index = matrixRow[i];
		if (v_index != -1)
		{
			int aggV_index = local_v2v[v_index];
			int measure = sub_g.verticesValue[v_index].measure;
			aggVList[aggV_index].measure += measure;
		}
	}
	gettimeofday(&end, NULL);
	threadTime[tid] = GetTime(end, start);	// record work time
}

void VM_thread(int tid, int startPos, int endPos, std::vector<AggVertex>& res_aggVList,
 std::vector<AggVertex>& aggVList,
	std::vector<int>& local_v2v, std::vector<long long>& threadTime)
{
	struct timeval start, end;
	gettimeofday(&start, NULL);
	for (int nid = startPos; nid < endPos; nid++)
	{
		int agg_nid = local_v2v[nid];
		int measure = aggVList[nid].measure;
		res_aggVList[agg_nid].measure += measure;
	}
	gettimeofday(&end, NULL);
	threadTime[tid] = GetTime(end, start);	// record work time
}

void EM_thread(int tid, int startPos, int endPos, std::vector<AggEdge>& res_aggEList,
 std::vector<AggEdge>& aggEList, std::vector<int>& local_e2e, std::vector<long long>& threadTime)
{
	struct timeval start, end;
	gettimeofday(&start, NULL);
	for (int nid = startPos; nid < endPos; nid++)
	{
		int agg_nid = local_e2e[nid];
		int weight = aggEList[nid].weight;
		res_aggEList[agg_nid].weight += weight;

	}
	gettimeofday(&end, NULL);
	threadTime[tid] = GetTime(end, start);	// record work time
}

void GraphEntityPackingEdgeOrigin(CubeAggGraph& g, int threadNum, std::vector<int>& v2v,
	std::vector<std::vector<int>>& matrix_edge)
{
	int avg_eSize = (g.eCount + threadNum - 1) / threadNum;
	for (int i = 0; i < matrix_edge.size(); i++)
	{
		matrix_edge[i].resize(avg_eSize, -1);
	}
	int row = 0, col = 0;
	for (int i = 0; i < g.eCount; i++)
	{
		matrix_edge[row][col++] = i;
		if (col == matrix_edge[row].size())
		{
			col = 0;
			row++;
		}
	}
}

void GraphEntityPackingEdge(CubeAggGraph& g, int threadNum, std::vector<int>& v2v,
	std::vector<std::vector<int>>& matrix_edge)
{
	struct mapImf
	{
		int aggE_index = 0;
		int TA = 0;
		std::vector<int> E_index;	//aggE_index to E_index
	};
	std::vector<mapImf> aggEdgeImfList;
	int aggCount = 0;
	// calc TA for all possible E_id
	std::unordered_map<long long int, mapImf> mapList;
	for (int i = 0; i < g.eCount; i++)
	{
		// get agg dimension
		long long int hashKeyEdge = CubeAgg<CubeAggVertexValue, int>::GetHashKeyAggEdge(v2v, g.eList[i].src, g.eList[i].dst);

		// if not exist in mapList
		if (mapList.find(hashKeyEdge) == mapList.end())
		{
			aggCount++;
			//mapImf tmp;
			mapList[hashKeyEdge] = mapImf();
			mapList[hashKeyEdge].aggE_index = aggCount - 1;
		}
		mapList[hashKeyEdge].TA++;
		mapList[hashKeyEdge].E_index.push_back(i);
	}
	// tranform the map result into vector
	aggEdgeImfList = std::vector<mapImf>(aggCount);
	for (auto it = mapList.begin(); it != mapList.end(); it++)
	{
		aggEdgeImfList[it->second.aggE_index].aggE_index = it->second.aggE_index;
		aggEdgeImfList[it->second.aggE_index].TA = it->second.TA;
		aggEdgeImfList[it->second.aggE_index].E_index = it->second.E_index;
	}

	int cc_threshold = (g.eCount + threadNum - 1) / threadNum; // |E|/s
	int q = 0;  // save the agg edge count with TA >= cc_threshold
	int maxTA = 0;  // save the maxTA
	std::vector<int> cc_Elist_index;  // save the agg edge index with TA >= cc_threshold
	std::vector<int> rest_Elist_index;  // save the rest agg edge index which TA < cc_threshold
	for (int i = 0; i < aggEdgeImfList.size(); i++)	// create cc_Vlist_index and rest_Vlist_index
	{
		if (aggEdgeImfList[i].TA >= cc_threshold)
		{
			q++;
			maxTA = aggEdgeImfList[i].TA > maxTA ? aggEdgeImfList[i].TA : maxTA;
			cc_Elist_index.push_back(i);
		}
		else
		{
			rest_Elist_index.push_back(i);
		}
	}

	// allocate memory for matrix_vertex with maxTA columns for each row,
	// do not forget to delete this memory after aggregation.
	for (int i = 0; i < matrix_edge.size(); i++)
	{
		matrix_edge[i].resize(std::max(maxTA, cc_threshold), -1);
	}

	// fill matrix_edge with cc_Elist_index
	for (int i = 0; i < q; i++)
	{
		int agg_index = cc_Elist_index[i];
		// fill a line with edge index which share the same aggE_key
		for (int j = 0; j < aggEdgeImfList[agg_index].E_index.size(); j++)
		{
			int e_index = aggEdgeImfList[agg_index].E_index[j];
			matrix_edge[i][j] = e_index;
		}
	}

	// fill matrix_edge with rest_Elist_index
	int i = q, j = 0;
	for (int k = 0; k < rest_Elist_index.size(); k++)
	{
		int agg_index = rest_Elist_index[k];
		// fill each line to the size = cc_threshold
		for (int l = 0; l < aggEdgeImfList[agg_index].E_index.size(); l++)
		{
			int e_index = aggEdgeImfList[agg_index].E_index[l];
			matrix_edge[i][j] = e_index;
			if (++j >= cc_threshold)  // line is full
			{
				i++;
				j = 0;
			}
		}
	}
}

int GetMergeCycleVertexPacking(CubeAggGraph& g, int threadNum, std::vector<int> aggDimension,
	std::vector<std::vector<int>>& matrix_vertex)
{
	//record each wasted MergeCycle, which will be merged again in extra MergeCycle
	std::unordered_map<std::string, int> extraMergeCycle;
	int wastedCount = 0;

	for (int j = 0; j < matrix_vertex[0].size(); j++)	// traverse each MergeCycle
	{
		std::unordered_map<std::string, int> tmp;
		for (int i = 0; i < matrix_vertex.size(); i++)
		{
			int vertex_index = matrix_vertex[i][j];
			if (vertex_index == -1)	continue;

			// get agg dimension
			std::string hashKeyVertex = CubeAgg<CubeAggVertexValue, int>::GetHashKeyAggVertex(aggDimension,
				g.verticesValue[vertex_index].dimension, g.verticesValue[vertex_index].dimension_ptr);

			if (tmp.find(hashKeyVertex) == tmp.end())	// not exist
			{
				tmp[hashKeyVertex] = 1;
			}
			else	// exist, save in extraMergeCycle
			{
				extraMergeCycle[hashKeyVertex]++;
				wastedCount++;
			}
		}
	}
	// get max extra merge cycle
	int maxWasted = 0;
	for (auto i = extraMergeCycle.begin(); i != extraMergeCycle.end(); i++)
	{
		if (i->second > maxWasted)	maxWasted = i->second;
	}
	int avg = (wastedCount + matrix_vertex.size() - 1) / matrix_vertex.size();

	return std::max(maxWasted, avg) + matrix_vertex[0].size();
}

int GetMergeCycleEdgePacking(CubeAggGraph& g, int threadNum, std::vector<int>& v2v,
	std::vector<std::vector<int>>& matrix_edge)
{
	//record each wasted MergeCycle, which will be merged again in extra MergeCycle
	std::unordered_map<long long, int> extraMergeCycle;
	int wastedCount = 0;

	for (int j = 0; j < matrix_edge[0].size(); j++)	// traverse each MergeCycle
	{
		std::unordered_map<long long, int> tmp;
		for (int i = 0; i < matrix_edge.size(); i++)
		{
			int edge_index = matrix_edge[i][j];
			if (edge_index == -1)	continue;

			// get hashKey
			long long int hashKeyEdge = CubeAgg<CubeAggVertexValue, int>::GetHashKeyAggEdge(v2v, g.eList[edge_index].src, g.eList[edge_index].dst);
			if (tmp.find(hashKeyEdge) == tmp.end())	// not exist
			{
				tmp[hashKeyEdge] = 1;
			}
			else	// exist, save in extraMergeCycle
			{
				extraMergeCycle[hashKeyEdge]++;
				wastedCount++;
			}
		}
	}
	// get max extra merge cycle
	int maxWasted = 0;
	for (auto i = extraMergeCycle.begin(); i != extraMergeCycle.end(); i++)
	{
		if (i->second > maxWasted)	maxWasted = i->second;
	}
	int avg = (wastedCount + matrix_edge.size() - 1) / matrix_edge.size();

	return std::max(maxWasted, avg) + matrix_edge[0].size();
}

void GraphEntityPackingVertexOrigin(CubeAggGraph& g, int threadNum, std::vector<int> aggDimension,
	std::vector<std::vector<int>>& matrix_vertex)
{
	int avg_v_size = (g.vCount + threadNum - 1) / threadNum;
	for (int i = 0; i < matrix_vertex.size(); i++)
	{
		matrix_vertex[i].resize(avg_v_size, -1);
	}
	int row = 0, col = 0;
	for (int i = 0; i < g.vCount; i++)
	{
		matrix_vertex[row][col++] = i;
		if (col == matrix_vertex[row].size())
		{
			col = 0;
			row++;
		}
	}
}

void GraphEntityPackingVertex(CubeAggGraph& g, int threadNum, std::vector<int> aggDimension,
	std::vector<std::vector<int>>& matrix_vertex)
{
	struct mapImf
	{
		int aggV_index = 0;
		int TA = 0;
		std::vector<int> V_index;
	};
	std::vector<mapImf> aggVertexImfList;
	int aggCount = 0;
	// calc TA for all possible V_id
	std::unordered_map<std::string, mapImf> mapList; //aggDimension -> <aggVid, TA>
	for (int i = 0; i < g.vCount; i++)
	{
		// get agg dimension
		std::string hashKeyVertex = CubeAgg<CubeAggVertexValue, int>::GetHashKeyAggVertex(aggDimension,
			g.verticesValue[i].dimension, g.verticesValue[i].dimension_ptr);

		// if not exist in mapList
		if (mapList.find(hashKeyVertex) == mapList.end())
		{
			aggCount++;
			mapImf tmp;
			mapList[hashKeyVertex] = tmp;
			mapList[hashKeyVertex].aggV_index = aggCount - 1;
		}
		mapList[hashKeyVertex].TA++;
		mapList[hashKeyVertex].V_index.push_back(i);
	}
	// tranform the map result into vector
	aggVertexImfList = std::vector<mapImf>(aggCount);
	for (auto it = mapList.begin(); it != mapList.end(); it++)
	{
		aggVertexImfList[it->second.aggV_index].aggV_index = it->second.aggV_index;
		aggVertexImfList[it->second.aggV_index].TA = it->second.TA;
		aggVertexImfList[it->second.aggV_index].V_index = it->second.V_index;
	}

	int cc_threshold = (g.vCount + threadNum - 1) / threadNum; // |V|/s
	int q = 0;  // save the agg vertex count with TA >= cc_threshold
	int maxTA = 0;  // save the maxTA
	std::vector<int> cc_Vlist_index;  // save the agg vertex index with TA >= cc_threshold
	std::vector<int> rest_Vlist_index;  // save the rest agg vertex index which TA < cc_threshold
	for (int i = 0; i < aggVertexImfList.size(); i++)	// create cc_Vlist_index and rest_Vlist_index
	{
		if (aggVertexImfList[i].TA >= cc_threshold)
		{
			q++;
			maxTA = aggVertexImfList[i].TA > maxTA ? aggVertexImfList[i].TA : maxTA;
			cc_Vlist_index.push_back(i);
		}
		else
		{
			rest_Vlist_index.push_back(i);
		}
	}

	// allocate memory for matrix_vertex with maxTA columns for each row,
	// do not forget to delete this memory after aggregation.
	for (int i = 0; i < matrix_vertex.size(); i++)
	{
		matrix_vertex[i].resize(std::max(maxTA, cc_threshold), -1);
	}

	// fill matrix_vertex with cc_Vlist_index
	for (int i = 0; i < q; i++)
	{
		int agg_index = cc_Vlist_index[i];
		// fill a line with vertex index which share the same aggDimension
		for (int j = 0; j < aggVertexImfList[agg_index].V_index.size(); j++)
		{
			int v_index = aggVertexImfList[agg_index].V_index[j];
			matrix_vertex[i][j] = v_index;
		}
	}

	// fill matrix_vertex with rest_Vlist_index
	int i = q, j = 0;
	for (int k = 0; k < rest_Vlist_index.size(); k++)
	{
		int agg_index = rest_Vlist_index[k];
		// fill each line to the size = cc_threshold
		for (int l = 0; l < aggVertexImfList[agg_index].V_index.size(); l++)
		{
			int v_index = aggVertexImfList[agg_index].V_index[l];
			matrix_vertex[i][j] = v_index;
			if (++j >= cc_threshold)  // line is full
			{
				i++;
				j = 0;
			}
		}
	}
}

template <typename VertexValueType, typename MessageValueType>
CubeAgg<VertexValueType, MessageValueType>::CubeAgg()
{
	this->map_d2v = std::unordered_map<std::string, int>();
	this->v2v = std::vector<int>();

	this->time_VC_C = 0;
	this->time_VC_M = 0;
	this->time_VM = 0;
	this->time_EC_C = 0;
	this->time_EC_M = 0;
	this->time_EM = 0;
	this->time = 0;

	this->fulltime_VC_C = 0;
	this->fulltime_VC_M = 0;
	this->fulltime_VM = 0;
	this->fulltime_EC_C = 0;
	this->fulltime_EC_M = 0;
	this->fulltime_EM = 0;
	this->fulltime = 0;

	this->sumMergeCycle_vertex = 0;
	this->sumMergeCycle_edge = 0;
	this->sumComputeCycle_vertex = 0;
	this->sumComputeCycle_edge = 0;
}

template <typename VertexValueType, typename MessageValueType>
int CubeAgg<VertexValueType, MessageValueType>::MSGApply(Graph<VertexValueType>& g,
	const std::vector<int>& initVSet, std::set<int>& activeVertice, const MessageSet<MessageValueType>& mSet)
{
	return 0;
}

template <typename VertexValueType, typename MessageValueType>
int CubeAgg<VertexValueType, MessageValueType>::MSGGenMerge(const Graph<VertexValueType>& g,
	const std::vector<int>& initVSet, const std::set<int>& activeVertice, MessageSet<MessageValueType>& mSet)
{
	return 0;
}

template <typename VertexValueType, typename MessageValueType>
void CubeAgg<VertexValueType, MessageValueType>::Init(int vCount, int eCount, int numOfInitV)
{
	this->totalVValuesCount = ((vCount * sizeof(CubeAggVertexValue))
		+ sizeof(VertexValueType) - 1) / sizeof(VertexValueType) * 10;
	this->totalMValuesCount = ((std::max((int)(eCount * sizeof(AggEdge)), (int)(vCount * sizeof(AggVertex))) + 2 * sizeof(long long) + sizeof(int))
		+ sizeof(MessageValueType) - 1) / sizeof(MessageValueType) * 1;
}

template <typename VertexValueType, typename MessageValueType>
void CubeAgg<VertexValueType, MessageValueType>::GraphInit(Graph<VertexValueType>& g, std::set<int>& activeVertices,
	const std::vector<int>& initVList)
{
}

template <typename VertexValueType, typename MessageValueType>
void CubeAgg<VertexValueType, MessageValueType>::Deploy(int vCount, int eCount, int numOfInitV)
{
}

template <typename VertexValueType, typename MessageValueType>
void CubeAgg<VertexValueType, MessageValueType>::Free()
{
}

template <typename VertexValueType, typename MessageValueType>
void CubeAgg<VertexValueType, MessageValueType>::MergeGraph(Graph<VertexValueType>& g, const std::vector<Graph<VertexValueType>>& subGSet,
	std::set<int>& activeVertices, const std::vector<std::set<int>>& activeVerticeSet,
	const std::vector<int>& initVList)
{
}

template <typename VertexValueType, typename MessageValueType>
int CubeAgg<VertexValueType, MessageValueType>::MSGGenMerge_array(int vCount, int eCount, const Vertex* vSet, const Edge* eSet, int numOfInitV, const int* initVSet, const VertexValueType* vValues, MessageValueType* mValues)
{
	//const Vertex*				vSet: vSet
	//const Edge*				eSet: eSet
	//const int*				initVset: type + s_vCount + s_eCount + aggDimension
	//const VertexValueType*	vValues: attrSet measure / v2v
	//MessageValueType*			mValues: maxTime_1 + maxTime_2 + mergeCycle + aggCount + AggVertex Set / AggEdge Set

	if (*(int*)initVSet == 0)	//vertex compute
	{
		// input and copy data
		CubeAggGraph sub_G;
		std::vector<AggVertex> aggVList;
		long long maxTimeCompute = 0;
		long long maxTimeMerge = 0;
		int mergeCycle = 0;
		std::vector<int> aggDimension(this->dCount);
		int v_count;

		// create sub_G
		v_count = *(int*)((long long)initVSet + sizeof(int));
		sub_G.vCount = v_count;
		int CubeAggValueSize = sizeof(CubeAggVertexValue);
		for (int i = 0; i < v_count; i++)
		{
			sub_G.vList.push_back(vSet[i]);
			CubeAggVertexValue tmp = *(CubeAggVertexValue*)((long long)vValues + i * CubeAggValueSize);
			sub_G.verticesValue.push_back(tmp);
		}
		// create aggDimension
		int offset = 3 * sizeof(int);	//offset of type
		for (int i = 0; i < this->dCount; i++)
		{
			int d = *(int*)((long long)initVSet + i * sizeof(int) + offset);
			aggDimension[i] = d;
		}

		this->VC_threadBlock(0, this->currentThread, sub_G, aggVList, aggDimension, maxTimeCompute, maxTimeMerge, mergeCycle);

		// output: maxTime_1 maxTime_2 mergeCycle aggCount aggV1 aggV2 ...
		*(long long*)mValues = maxTimeCompute;
		*(long long*)((long long)mValues + sizeof(long long)) = maxTimeMerge;
		*(int*)((long long)mValues + 2 * sizeof(long long)) = mergeCycle;
		*(int*)((long long)mValues + 2 * sizeof(long long) + sizeof(int)) = aggVList.size();
		offset = 2 * sizeof(long long) + 2 * sizeof(int);
		int aggV_size = sizeof(AggVertex);
		for (int i = 0; i < aggVList.size(); i++)
		{
			// int int [10] [10] ... [10]
			long long address = (long long)mValues + i * aggV_size + offset;
			*(AggVertex*)address = aggVList[i];
		}
	}
	else if (*(int*)initVSet == 1)	//edge compute
	{
		// input and copy data
		CubeAggGraph sub_G;
		std::vector<AggEdge> aggEList;
		long long maxTime_compute;
		long long maxTime_merge;
		int mergeCycle = 0;
		std::vector<int> v2v;
		int e_count;

		// create sub_G
		e_count = *(int*)((long long)initVSet + 2 * sizeof(int));

		sub_G.eCount = e_count;
		for (int i = 0; i < e_count; i++)
		{
			sub_G.eList.push_back(eSet[i]);
		}
		// create v2v
		for (int i = 0; i < vCount; i++)
		{
			v2v.push_back(*(int*)((long long)vValues + i * sizeof(int)));
		}

		this->EC_threadBlock(0, this->currentThread, sub_G, aggEList, v2v, maxTime_compute, maxTime_merge, mergeCycle);

		// output: maxTime_1 maxTime_2 mergeCycle aggCount aggE1 aggE2 ...
		*(long long*)mValues = maxTime_compute;
		*(long long*)((long long)mValues + sizeof(long long)) = maxTime_merge;
		*(int*)((long long)mValues + 2 * sizeof(long long)) = mergeCycle;
		*(int*)((long long)mValues + 2 * sizeof(long long) + sizeof(int)) = aggEList.size();
		int offset = 2 * sizeof(long long) + 2 * sizeof(int);
		for (int i = 0; i < aggEList.size(); i++)
		{
			*(AggEdge*)((long long)mValues + i * sizeof(AggEdge) + offset) = aggEList[i];
		}
	}
	else
	{
		std::cout << "error value of initVSet[0]!" << std::endl;
		throw std::runtime_error("error value of initVSet[0]!");
		return 0;
	}
	return 1;
}

template <typename VertexValueType, typename MessageValueType>
int CubeAgg<VertexValueType, MessageValueType>::MSGApply_array(int vCount, int eCount, Vertex* vSet, int numOfInitV, const int* initVSet, VertexValueType* vValues, MessageValueType* mValues)
{
	return 0;
}

//this function is to get aggVList, which is a intermediate vertex subGraph, whose aggVertex has only the dimension and measure
template <typename VertexValueType, typename MessageValueType>
void CubeAgg<VertexValueType, MessageValueType>::VC_threadBlock(int tid, int threadNum, CubeAggGraph& sub_g, std::vector<AggVertex>& aggVList,
	std::vector<int> aggDimension, long long& maxTimeCompute, long long& maxTime_merge, int& mergeCycle)
{
	std::unordered_map<std::string, int> hashTable;  // used in pre_build
	std::vector<std::thread> threads(threadNum);	// threads
	int aggCount = 0;	// agg vertex count

	// create schedule matrix
	std::vector<std::vector<int>> matrix_vertex_compute(threadNum);
	std::vector<std::vector<int>> matrix_vertex_merge(threadNum);
	// packing for compute cycle
	GraphEntityPackingVertexOrigin(sub_g, threadNum, aggDimension, matrix_vertex_compute);
	// packing for merge cycle
	GraphEntityPackingVertex(sub_g, threadNum, aggDimension, matrix_vertex_merge);
	// get merge cycle
	mergeCycle = GetMergeCycleVertexPacking(sub_g, threadNum, aggDimension, matrix_vertex_merge);

	// VC Compute begin
	std::vector<std::string> v2hashKey(sub_g.vCount);
	std::vector<long long> threadTimeCompute(threadNum);

	for (int i = 0; i < threadNum; i++)	// create threads
	{
		threads[i] = std::thread(VC_compute_thread, i,
			std::ref(sub_g.verticesValue), std::ref(aggDimension), std::ref(v2hashKey),
			std::ref(matrix_vertex_compute[i]), std::ref(threadTimeCompute));
	}
	for (int i = 0; i < threadNum; i++)
	{
		threads[i].join();
	}

	// return max thread work time
	maxTimeCompute = *std::max_element(threadTimeCompute.begin(), threadTimeCompute.end());
	// VC Compute end

	// VC Merge begin
	std::vector<int> local_v2v(sub_g.vCount);
	std::vector<long long> threadTimeMerge(threadNum);

	// pre build start
	for (int i = 0; i < sub_g.vCount; i++)
	{
		// if not exist in hashTable
		if (hashTable.find(v2hashKey[i]) == hashTable.end())
		{
			aggCount++;
			hashTable[v2hashKey[i]] = aggCount - 1;
			char v_dimension[100];
			for (int v = 0; v < 100; v++)
				v_dimension[v] = 0;
			for (int k = 0; k < aggDimension.size(); k++)
			{
				if (aggDimension.at(k) != 0)
				{
					memcpy(v_dimension + 10 * k, sub_g.verticesValue[i].dimension_ptr[k], 10);
				}
			}
			AggVertex tmp(v_dimension, 0);
			aggVList.push_back(tmp);
		}
		// record imf in VC_merge_thread
		local_v2v[i] = hashTable[v2hashKey[i]];
	}
	// pre build end

	for (int i = 0; i < threadNum; i++)	// create threads
	{
		threads[i] = std::thread(VC_merge_thread, i, std::ref(sub_g), std::ref(aggVList), std::ref(local_v2v),
			std::ref(matrix_vertex_merge[i]), std::ref(threadTimeMerge));
	}
	for (int i = 0; i < threadNum; i++)
	{
		threads[i].join();
	}

	// return max thread work time
	maxTime_merge = *std::max_element(threadTimeMerge.begin(), threadTimeMerge.end());

	// release matrix_vertex memory
	for (int i = 0; i < matrix_vertex_compute.size(); i++)	std::vector<int>().swap(matrix_vertex_compute[i]);
	for (int i = 0; i < matrix_vertex_merge.size(); i++)	std::vector<int>().swap(matrix_vertex_merge[i]);
}

template <typename VertexValueType, typename MessageValueType>
void CubeAgg<VertexValueType, MessageValueType>::EC_threadBlock(int tid, int threadNum, CubeAggGraph& sub_g, std::vector<AggEdge>& aggEList,
	std::vector<int>& v2v, long long& maxTimeCompute, long long& maxTimeMerge, int& mergeCycle)
{
	std::unordered_map<long long int, int> hashTable;  // used in pre_build
	int aggCount = 0;	// agg vertex count
	std::vector<std::thread> threads(threadNum);	//threads

	// create schedule matrix
	std::vector<std::vector<int>> matrix_edge_compute(threadNum);
	std::vector<std::vector<int>> matrix_edge_merge(threadNum);
	// packing for compute cycle
	GraphEntityPackingEdgeOrigin(sub_g, threadNum, v2v, matrix_edge_compute);
	// packing for merge cycle
	GraphEntityPackingEdge(sub_g, threadNum, v2v, matrix_edge_merge);
	// get mergecycle
	mergeCycle = GetMergeCycleEdgePacking(sub_g, threadNum, v2v, matrix_edge_merge);

	//EC compute begin
	std::vector<long long> threadTimeCompute(threadNum);
	std::vector<long long> e2hashKey(sub_g.eCount);

	for (int i = 0; i < threadNum; i++)	// create threads
	{
		threads[i] = std::thread(EC_compute_thread, i,
			std::ref(sub_g.eList), std::ref(v2v), std::ref(e2hashKey),
			std::ref(matrix_edge_compute[i]), std::ref(threadTimeCompute));
	}
	for (int i = 0; i < threadNum; i++)
	{
		threads[i].join();
	}

	// return max thread work time
	maxTimeCompute = *std::max_element(threadTimeCompute.begin(), threadTimeCompute.end());
	//EC compute end

	//EC merge begin
	std::vector<int> local_e2e(sub_g.eCount);
	std::vector<long long> threadTimeMerge(threadNum);

	// pre build start
	for (int i = 0; i < sub_g.eCount; i++)
	{
		int src = sub_g.eList[i].src;
		int dst = sub_g.eList[i].dst;
		int agg_src = v2v[src];
		int agg_dst = v2v[dst];
		// if not exist in hashTable
		if (hashTable.find(e2hashKey[i]) == hashTable.end())
		{
			aggCount++;
			hashTable[e2hashKey[i]] = aggCount - 1;
			aggEList.emplace_back(agg_src, agg_dst);
		}
		// record imf in EC_merge_thread
		local_e2e[i] = hashTable[e2hashKey[i]];
	}
	// pre build end

	for (int i = 0; i < threadNum; i++)	// create threads
	{
		threads[i] = std::thread(EC_merge_thread, i, std::ref(sub_g), std::ref(aggEList),
			std::ref(local_e2e), std::ref(matrix_edge_merge[i]), std::ref(threadTimeMerge));
	}
	for (int i = 0; i < threadNum; i++)
	{
		threads[i].join();
	}

	// return max thread work time
	maxTimeMerge = *std::max_element(threadTimeMerge.begin(), threadTimeMerge.end());
	//EC merge end

	// release matrix_edge memory
	for (int i = 0; i < matrix_edge_compute.size(); i++)	std::vector<int>().swap(matrix_edge_compute[i]);
	for (int i = 0; i < matrix_edge_merge.size(); i++)	std::vector<int>().swap(matrix_edge_merge[i]);
}

//This function be called only by non-middleware program
template <typename VertexValueType, typename MessageValueType>
void CubeAgg<VertexValueType, MessageValueType>::VertexCompute(std::vector<CubeAggGraph>& subGSet,
	std::vector<std::vector<AggVertex>>& aggVListSet,
	std::vector<int> aggDimension)
{
	std::vector<long long> maxTimeCompute(this->currentPartition);
	std::vector<long long> maxTimeMerge(this->currentPartition);
	std::vector<int> mergeCycle(this->currentPartition);
	std::vector<std::thread> threads(this->currentPartition);

	for (int i = 0; i < this->currentPartition; i++)
	{
		threads[i] = std::thread(VC_threadBlock, i, this->currentThread, std::ref(subGSet[i]), std::ref(aggVListSet[i]),
			aggDimension, std::ref(maxTimeCompute[i]), std::ref(maxTimeMerge[i]), std::ref(mergeCycle[i]));
	}
	for (int i = 0; i < this->currentPartition; i++)
	{
		threads[i].join();
	}
	this->time_VC_C = *std::max_element(maxTimeCompute.begin(), maxTimeCompute.end());
	this->time_VC_M = *std::max_element(maxTimeMerge.begin(), maxTimeMerge.end());
	this->mergeCycle_vertex = *std::max_element(mergeCycle.begin(), mergeCycle.end());
}

//This function be called only by non-middleware program
template <typename VertexValueType, typename MessageValueType>
void CubeAgg<VertexValueType, MessageValueType>::EdgeCompute(std::vector<CubeAggGraph>& subGSet,
	std::vector<std::vector<AggEdge>>& aggEListSet)
{
	std::vector<long long> maxTimeCompute(this->currentPartition);
	std::vector<long long> maxTimeMerge(this->currentPartition);
	std::vector<int> mergeCycle(this->currentPartition);
	std::vector<std::thread> threads(this->currentPartition);

	for (int i = 0; i < this->currentPartition; i++)
	{
		threads[i] = std::thread(EC_threadBlock, i, this->currentThread, std::ref(subGSet[i]), std::ref(aggEListSet[i]),
			std::ref(this->v2v), std::ref(maxTimeCompute[i]), std::ref(maxTimeMerge[i]), std::ref(mergeCycle[i]));
	}
	for (int i = 0; i < this->currentPartition; i++)
	{
		threads[i].join();
	}
	this->time_EC_C = *std::max_element(maxTimeCompute.begin(), maxTimeCompute.end());
	this->time_EC_M = *std::max_element(maxTimeMerge.begin(), maxTimeMerge.end());
	this->mergeCycle_edge = *std::max_element(mergeCycle.begin(), mergeCycle.end());

}

template <typename VertexValueType, typename MessageValueType>
void CubeAgg<VertexValueType, MessageValueType>::EdgeMerge(CubeAggGraph& g, std::vector<std::vector<AggEdge>>& aggEListSet)
{
	std::unordered_map<long long int, int> hashTable;
	std::vector<AggEdge> resultAggEList;
	int aggCount = 0;

	// used in VM_thread
	std::vector<std::vector<int>> local_e2e(aggEListSet.size());
	for (int i = 0; i < aggEListSet.size(); i++)
	{
		local_e2e[i] = std::vector<int>(aggEListSet[i].size());
	}

	// pre build start
	for (int i = 0; i < aggEListSet.size(); i++)
	{
		for (int j = 0; j < aggEListSet[i].size(); j++)
		{
			int src = aggEListSet[i][j].src;
			int dst = aggEListSet[i][j].dst;
			long long hashKeyEdge = GetHashKeyMergeEdge(src, dst);
			if (hashTable.find(hashKeyEdge) == hashTable.end())  // not exist in hashTable
			{
				aggCount++;
				hashTable[hashKeyEdge] = aggCount - 1;
				resultAggEList.emplace_back(src, dst);
				g.vList[src].outDegree++;
				g.vList[dst].inDegree++;
			}
			local_e2e[i][j] = hashTable[hashKeyEdge];
		}
	}
	// pre build end

	int threadNum = this->currentThread;

	std::vector<std::thread> threads(threadNum);

	// sequently merge each aggEList
	for (int i = 0; i < this->currentPartition; i++)
	{
		int e_size = aggEListSet[i].size();
		int e_avg = (e_size + threadNum - 1) / threadNum;
		std::vector<long long> threadTime(threadNum);

		for (int j = 0; j < threadNum; j++)	// create threads
		{
			int startPos = j * e_avg;
			int endPos = std::min((j + 1) * e_avg, e_size);
			threads[j] = std::thread(EM_thread, j, startPos, endPos, std::ref(resultAggEList),
				std::ref(aggEListSet[i]), std::ref(local_e2e[i]), std::ref(threadTime));
		}
		for (int i = 0; i < threadNum; i++)
		{
			threads[i].join();
		}
		this->time_EM += *std::max_element(threadTime.begin(), threadTime.end());
	}

	// transfrom aggVList into res graph
	g.eCount = aggCount;
	g.eList = std::vector<Edge>(aggCount);
	for (int i = 0; i < resultAggEList.size(); i++)
	{
		int src = resultAggEList[i].src;
		int dst = resultAggEList[i].dst;
		int weight = resultAggEList[i].weight;
		g.eList[i].src = src;
		g.eList[i].dst = dst;
		g.eList[i].weight = weight;
	}
	return;
}

template <typename VertexValueType, typename MessageValueType>
void CubeAgg<VertexValueType, MessageValueType>::VertexMerge(CubeAggGraph& g, std::vector<std::vector<AggVertex>>& aggVListSet,
	std::vector<int> aggDimension)
{
	std::unordered_map<std::string, int> hashTable;
	std::vector<AggVertex> resultAggVList;
	int aggCount = 0;

	// used in VM_thread
	std::vector<std::vector<int>> local_v2v(aggVListSet.size());
	for (int i = 0; i < aggVListSet.size(); i++)
	{
		local_v2v[i] = std::vector<int>(aggVListSet[i].size());
	}

	// pre build start
	for (int i = 0; i < aggVListSet.size(); i++)
	{
		for (int j = 0; j < aggVListSet[i].size(); j++)
		{
			std::string hashKeyVertex = GetHashKeyMergeVertex(aggVListSet[i][j].dimension, 
				aggVListSet[i][j].dimension_ptr);

			if (hashTable.find(hashKeyVertex) == hashTable.end())  // not exist in hashTable
			{
				aggCount++;
				hashTable[hashKeyVertex] = aggCount - 1;
				resultAggVList.push_back(aggVListSet[i][j]);
			}
			local_v2v[i][j] = hashTable[hashKeyVertex];
		}
	}
	// pre build end

	int threadNum = this->currentThread;

	std::vector<std::thread> threads(threadNum);

	// sequently merge each aggVList
	for (int i = 0; i < this->currentPartition; i++)
	{
		int v_size = aggVListSet[i].size();
		int v_avg = (v_size + threadNum - 1) / threadNum;

		std::vector<long long> threadTime(threadNum);

		for (int j = 0; j < threadNum; j++)	// create threads
		{
			int startPos = j * v_avg;
			int endPos = std::min((j + 1) * v_avg, v_size);
			threads[j] = std::thread(VM_thread, j, startPos, endPos, std::ref(resultAggVList), std::ref(aggVListSet[i]),
				std::ref(local_v2v[i]), std::ref(threadTime));
		}
		for (int i = 0; i < threadNum; i++)
		{
			threads[i].join();
		}
		this->time_VM += *std::max_element(threadTime.begin(), threadTime.end());
	}

	// transfrom aggVList into res graph
	g.vCount = aggCount;
	g.vList = std::vector<Vertex>(aggCount);
	g.verticesValue = std::vector<CubeAggVertexValue>(aggCount);
	for (int i = 0; i < resultAggVList.size(); i++)
	{
		std::string hashKeyVertex = GetHashKeyMergeVertex(resultAggVList[i].dimension, 
			resultAggVList[i].dimension_ptr);

		int agg_vertexID = this->map_d2v[hashKeyVertex];
		g.vList[agg_vertexID].vertexID = agg_vertexID;
		for (int v = 0; v < 10; v++)
		{
			*g.verticesValue[agg_vertexID].dimension_ptr[v] = *resultAggVList[i].dimension_ptr[v];
		}
		g.verticesValue[agg_vertexID].measure = resultAggVList[i].measure;
	}
	return;
}

template <typename VertexValueType, typename MessageValueType>
int CubeAgg<VertexValueType, MessageValueType>::GetMaxValueOfMap(const std::unordered_map<long long int, int>& mp)
{
	int max = -0x7fffffff;
	for (auto i = mp.begin(); i != mp.end(); i++)
	{
		if (i->second > max)	max = i->second;
	}
	return max;
}

template <typename VertexValueType, typename MessageValueType>
void CubeAgg<VertexValueType, MessageValueType>::GetOptimalMergeCycle
(const CubeAggGraph& g, std::vector<int> aggDimension, int partitionNum, int threadNum)
{
	std::unordered_map<std::string, int> map_d2v; // vertexKey to agg_vertexID
	std::vector<int> v2v(g.vCount,-1);	//vertexID to agg_vertexID
	int aggVCount = 0;

	// create v2v
	for (int i = 0; i < g.vCount; i++)
	{
		std::string hashKeyVertex = GetHashKeyAggVertex(aggDimension, g.verticesValue[i].dimension, g.verticesValue[i].dimension_ptr);

		if (map_d2v.find(hashKeyVertex) == map_d2v.end())
		{
			aggVCount++;
			map_d2v[hashKeyVertex] = aggVCount - 1;
		}
		int vertexID = g.vList[i].vertexID;
		v2v[vertexID] = map_d2v[hashKeyVertex];
	}

	// get vertex mergeCycle
	std::unordered_map<std::string, int> d2v;
	for (int i = 0; i < g.vCount; i++)
	{
		std::string hashKeyVertex = GetHashKeyAggVertex(aggDimension, g.verticesValue[i].dimension, g.verticesValue[i].dimension_ptr);

		if (d2v.find(hashKeyVertex) == d2v.end())
		{
			d2v[hashKeyVertex] = 0;
		}
		d2v[hashKeyVertex] += 1;
	}

	// get maxTA_edge
	std::unordered_map<long long, int> d2e;
	for (int i = 0; i < g.eCount; i++)
	{
		long long int hashKeyEdge = GetHashKeyAggEdge(v2v, g.eList[i].src, g.eList[i].dst);
		if (d2e.find(hashKeyEdge) == d2e.end())
		{
			d2e[hashKeyEdge] = 0;
		}
		d2e[hashKeyEdge] += 1;
	}

	int maxTA_vertex = 0;
	for (auto it = d2v.begin(); it != d2v.end(); it++)
	{
		maxTA_vertex = std::max(maxTA_vertex, it->second);
	}
	int maxTA_edge = 0;
	for (auto it = d2e.begin(); it != d2e.end(); it++)
	{
		maxTA_edge = std::max(maxTA_edge, it->second);
	}

	int AVG_V = (((g.vCount + partitionNum - 1) / partitionNum) + threadNum - 1) / threadNum;
	int AVG_E = (((g.eCount + partitionNum - 1) / partitionNum) + threadNum - 1) / threadNum;
	int OMCV = std::max((maxTA_vertex + partitionNum - 1) / partitionNum, AVG_V);
	int OMCE = std::max((maxTA_edge + partitionNum - 1) / partitionNum, AVG_E);
	std::cout << "optimal mergeCycle: " << OMCV + OMCE << std::endl;
}

template <typename VertexValueType, typename MessageValueType>
void CubeAgg<VertexValueType, MessageValueType>::GetV2V(CubeAggGraph& g, std::vector<int> aggDimension)
{
	std::unordered_map<int, std::string> map_v2d;
	this->v2v.resize(g.vCount, -1);

	for (int i = 0; i < g.vCount; i++)
	{
		std::string hashKeyVertex = GetHashKeyAggVertex(aggDimension, g.verticesValue[i].dimension, g.verticesValue[i].dimension_ptr);
		
		int vertexID = g.vList[i].vertexID;
		map_v2d[vertexID] = hashKeyVertex;
	}
	int aggCount = 0;
	for (auto it = map_v2d.begin(); it != map_v2d.end(); it++)
	{
		if (this->map_d2v.find(it->second) == this->map_d2v.end())
		{
			this->map_d2v.insert(std::pair<std::string, int>(it->second, aggCount++));  // get d2v
			this->v2v[it->first] = aggCount - 1; // get v2v
		}
		else
		{
			this->v2v[it->first] = this->map_d2v[it->second];
		}
	}
	return;
}

template <typename VertexValueType, typename MessageValueType>
std::vector<CubeAggGraph> CubeAgg<VertexValueType, MessageValueType>::DivideGraph
(CubeAggGraph& g, int partitionCount, int threadCount, std::vector<int> aggDimension)
{
	std::vector<CubeAggGraph> res = std::vector<CubeAggGraph>();
	for (int j = 0; j < partitionCount; j++) res.push_back(CubeAggGraph());

	std::unordered_map<std::string, int> d2v;
	std::unordered_map<long long, int> d2e;
	std::vector<int> v2v(g.vCount);
	std::vector<int> e2e(g.eCount);
	int aggVertexCount = 0;
	int aggEdgeCount = 0;

	// get v2v
	for (int i = 0; i < g.vCount; i++)
	{
		std::string v_id = GetHashKeyAggVertex(aggDimension, g.verticesValue[i].dimension, g.verticesValue[i].dimension_ptr);
	
		if (d2v.find(v_id) == d2v.end())
		{
			d2v[v_id] = aggVertexCount++;
		}
		v2v[i] = d2v[v_id];
	}
	//get e2e
	for (int i = 0; i < g.eCount; i++)
	{
		int src = g.eList[i].src;
		int dst = g.eList[i].dst;
		long long e_id = (long long)v2v[src] * MAX_NODE_NUMBER + (long long)v2v[dst];
		if (d2e.find(e_id) == d2e.end())
		{
			d2e[e_id] = aggEdgeCount++;
		}
		e2e[i] = d2e[e_id];
	}
	// init insertPos
	std::vector<int> insertPosVertex(aggVertexCount);
	std::vector<int> insertPosEdge(aggEdgeCount);
	for (int i = 0; i < aggVertexCount; i++)
	{
		insertPosVertex[i] = i % partitionCount;
	}
	for (int i = 0; i < aggEdgeCount; i++)
	{
		insertPosEdge[i] = i % partitionCount;
	}

	//insert vertex
	for (int i = 0; i < g.vCount; i++)
	{
		int aggV_index = v2v[i];
		int pid = insertPosVertex[aggV_index];
		insertPosVertex[aggV_index] = (insertPosVertex[aggV_index] + 1) % partitionCount;

		res[pid].vCount++;
		res[pid].vList.push_back(g.vList[i]);
		res[pid].verticesValue.push_back(g.verticesValue[i]);
	}
	//insert edge
	for (int i = 0; i < g.eCount; i++)
	{
		int aggE_index = e2e[i];
		int pid = insertPosEdge[aggE_index];
		insertPosEdge[aggE_index] = (insertPosEdge[aggE_index] + 1) % partitionCount;

		res[pid].eCount++;
		res[pid].eList.push_back(g.eList[i]);
	}
	return res;
}

template <typename VertexValueType, typename MessageValueType>
void  CubeAgg<VertexValueType, MessageValueType>::AggInit()
{
	this->map_d2v = std::unordered_map<std::string, int>();
	this->v2v = std::vector<int>();

	this->time_VC_C = this->time_VC_M = this->time_VM = this->time_EC_C = this->time_EC_M = this->time_EM = this->time = 0;
	this->mergeCycle_vertex = this->mergeCycle_edge = 0;
}

template <typename VertexValueType, typename MessageValueType>
void  CubeAgg<VertexValueType, MessageValueType>::AggEnd()
{
	this->time = this->time_VC_C + this->time_VC_M + this->time_VM + this->time_EC_C + this->time_EC_M + this->time_EM;
	this->fulltime += this->time;
	this->fulltime_VC_C += this->time_VC_C;
	this->fulltime_VC_M += this->time_VC_M;
	this->fulltime_VM += this->time_VM;
	this->fulltime_EC_C += this->time_EC_C;
	this->fulltime_EC_M += this->time_EC_M;
	this->fulltime_EM += this->time_EM;

	this->sumMergeCycle_vertex += this->mergeCycle_vertex;
	this->sumMergeCycle_edge += this->mergeCycle_edge;
}

template <typename VertexValueType, typename MessageValueType>
std::string CubeAgg<VertexValueType, MessageValueType>::GetHashKeyAggVertex
(const std::vector<int>& aggDimension, const char dimension[100], char* const dimension_ptr[10])
{
	std::string hashKeyVertex("0");
	for (int d = 0; d < aggDimension.size(); d++)
	{
		if (aggDimension[d] != 0)
		{
			hashKeyVertex += "/";
			hashKeyVertex += dimension_ptr[d];
		}
	}
	return hashKeyVertex;
}

template <typename VertexValueType, typename MessageValueType>
std::string CubeAgg<VertexValueType, MessageValueType>::GetHashKeyMergeVertex
(const char dimension[100], char* const dimension_ptr[10])
{
	std::string hashKeyVertex("0");
	for (int d = 0; d < CubeAggVertexValue::max_dimension_count; d++)
	{
		if (*dimension_ptr[d] != 0)
		{
			hashKeyVertex += "/";
			hashKeyVertex += dimension_ptr[d];
		}
	}
	return hashKeyVertex;
}

template <typename VertexValueType, typename MessageValueType>
long long CubeAgg<VertexValueType, MessageValueType>::GetHashKeyAggEdge
(const std::vector<int>& v2v, int src, int dst)
{
	int agg_src = v2v[src];
	int agg_dst = v2v[dst];
	long long hashKeyEdge = (long long)agg_src * MAX_NODE_NUMBER + (long long)agg_dst;
	return hashKeyEdge;
}

template <typename VertexValueType, typename MessageValueType>
long long CubeAgg<VertexValueType, MessageValueType>::GetHashKeyMergeEdge
(int src, int dst)
{
	long long hashKeyEdge = (long long)src * MAX_NODE_NUMBER + (long long)dst;
	return hashKeyEdge;
}

template <typename VertexValueType, typename MessageValueType>
void CubeAgg<VertexValueType, MessageValueType>::ApplyD(CubeAggGraph& g, 
	const std::vector<std::vector<int>>& aggDimension, std::vector<CubeAggGraph>& agg_gSet, 
	const std::vector<int>& dgList, int cuboidID)
{
	int cuboidCount = 0;
	int d_len = aggDimension.at(0).size();

	// full materialization: 2^d cuboidCount
	while (cuboidCount < pow(2, d_len))
	{
		cuboidCount++;
		
		if (cuboidCount != cuboidID)	continue;

		//Test
		std::cout << "Start:" << clock() << std::endl;
		//Test end

		CubeAggGraph* gd = &g;

		AggInit();

		auto subGraphSet = this->DivideGraph((*gd), this->currentPartition, this->currentThread, aggDimension.at(cuboidCount - 1));		

		std::vector<CubeAggGraph> agg_subGraphSet(this->currentPartition);  // agg sub graph set
		std::vector<std::vector<AggVertex>> aggVList_set(this->currentPartition);  // intermediate agg sub graph set
		std::vector<std::vector<AggEdge>> aggEList_set(this->currentPartition);  // intermediate agg sub graph set

		// v compute
		VertexCompute(subGraphSet, aggVList_set, aggDimension[cuboidCount - 1]);

		//Test
		std::cout << "time_VC_C:" << this->time_VC_C << std::endl;
		std::cout << "time_VC_M:" << this->time_VC_M << std::endl;
		//Test end

		// get v2v
		GetV2V((*gd), aggDimension[cuboidCount - 1]);

		// v merge
		VertexMerge(agg_gSet[cuboidCount - 1], aggVList_set, aggDimension.at(cuboidCount - 1));

		//Test
		std::cout << "time_VM:" << this->time_VM << std::endl;
		//Test end

		// e compute
		EdgeCompute(subGraphSet, aggEList_set);

		//Test
		std::cout << "time_EC_C:" << this->time_EC_C << std::endl;
		std::cout << "time_EC_M:" << this->time_EC_M << std::endl;
		//Test end

		// e merge
		EdgeMerge(agg_gSet[cuboidCount - 1], aggEList_set);

		//Test
		std::cout << "time_EM:" << this->time_EM << std::endl;
		//Test end

		AggEnd();

		//Test
		std::cout << "time:" << this->time << std::endl;
		std::cout << "mergeCycle:" << this->mergeCycle_vertex + this->mergeCycle_edge << std::endl;
		//Test end

		//test
		this->GetOptimalMergeCycle((*gd), aggDimension[cuboidCount - 1], this->currentPartition, this->currentThread);
	}
}

template class CubeAgg<CubeAggVertexValue, int>;