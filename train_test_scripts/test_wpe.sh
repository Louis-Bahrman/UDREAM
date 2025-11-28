#!/bin/bash

#SBATCH --partition=P100
#SBATCH --array=0-7
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10

# #SBATCH --partition=CPU
# #SBATCH --array=0-35
# #SBATCH --cpus-per-task=10

hostname=`hostname`
if [[ ${hostname,,} == *louis* ]];
then 
    debug_config=" --config=config/debug.yaml"
else
    export PATH="~/miniconda3/bin:$PATH"
    eval "$(conda shell.bash hook)"
    conda activate hybrid_wssd
fi

cartesian_product=({ears,synth}+{nara,espnet}+3+{3,10,27}+{1,3,5})
cartesian_product=({ears,synth}+dnn_wpe_{bilstm_strong_100,bilstm_strong,bilstm_strong_5percent,bilstm_unsupervised}+3+10+3)
current_config="${cartesian_product[$SLURM_ARRAY_TASK_ID]}"

OLDIFS=$IFS
IFS="+"
set -- $current_config
IFS=$OLDIFS

dataset=$1
implementation=$2
delay=$3
taps=$4
iterations=$5

echo "dataset: $dataset"
echo "implementation: $implementation"
echo "delay: $delay"
echo "taps: $taps"
echo "iterations: $iterations"


# check if srun is available
if command -v srun 2>&1 >/dev/null
then 
  srun_if_available="srun "
else
  srun_if_available=""
fi
$srun_if_available python test_wpe.py --dataset $dataset --implementation $implementation --delay $delay --taps $taps --iterations $iterations
