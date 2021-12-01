#!/bin/bash


if [ $# != 7 ] ; then
  echo "Usage: ./mpi_CubeAggTest.sh vCount eCount dCount nodeCount perPartitionCount threadCount cuboidID"
  exit 1;
fi 

vCount=$1
eCount=$2
dCount=$3
nodeCount=$4
perPartitionCount=$5
threadCount=$6
cuboidID=$7

sbatch --nodes=$[$nodeCount+1] sbatch_CubeAgg.sh $vCount $eCount $dCount $nodeCount $perPartitionCount $threadCount $cuboidID