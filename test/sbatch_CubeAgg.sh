#!/bin/sh
##SBATCH -p CPU-Shorttime
##SBATCH -q qos_cpu_shorttime
#SBATCH -p test
#SBATCH -q testqos
#SBATCH -o exp.%j.out
#SBATCH -J CubeAgg
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40

echo Time is `date`
echo Directory is $PWD
echo This job runs on the following nodes:
echo $SLURM_JOB_NODELIST
echo This job has allocated $SLURM_JOB_CPUS_PER_NODE cpu cores.

vCount=$1
eCount=$2
dCount=$3
nodeCount=$4
perPartitionCount=$5
threadCount=$6
cuboidID=$7

#build workDir
mkdir exp_$SLURM_JOB_ID
cd exp_$SLURM_JOB_ID
cp ../testGraph*.txt ./

#transfrom the allocated nodes into array: nodelist
scontrol show hostname $SLURM_JOB_NODELIST | perl -ne 'chomb; print "$_"x1' > nodelist.txt
nodelist=()
while read line
do
nodelist[${#nodelist[*]}]="$line"
done < nodelist.txt


echo "mission start"

#release shared memory and create server
for((i=0;i<${#nodelist[*]};i++))
do
ssh ${nodelist[i]} > /dev/null 2>&1 << eeooff
../ipcrm_32.sh > ${nodelist[i]}_ipcrm

for (( i = 0; i < $perPartitionCount ; i ++ ))
do
    nohup ../../build/bin/srv_UtilServerTest_CubeAgg $vCount $eCount $dCount $i $threadCount &
    echo "create server"
done

exit
eeooff
done
echo "ipcrm and create server OK"

#main work
mpirun -n $[$nodeCount+1] -ppn 1 ../../build/bin/srv_UtilClientTest_CubeAgg $vCount $eCount $dCount $nodeCount $perPartitionCount $threadCount $cuboidID
echo "mission completed"

#release shared memory
for((i=0;i<${#nodelist[*]};i++))
do
ssh ${nodelist[i]} > /dev/null 2>&1 << eeooff
../ipcrm_32.sh >> ${nodelist[i]}_ipcrm
exit
eeooff
done
echo "ipcrm OK"

cd ..
rm -rf exp_$SLURM_JOB_ID
mv exp.$SLURM_JOB_ID.out ${SLURM_JOB_ID}:v-${vCount}_${nodeCount}_${perPartitionCount}_${threadCount}.out

