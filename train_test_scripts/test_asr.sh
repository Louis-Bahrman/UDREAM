#!/bin/bash

#SBATCH --partition=P100
#SBATCH --gpus=1
#SBATCH --array=0-8
#SBATCH --cpus-per-task=10

hostname=`hostname`
if [[ ${hostname,,} == *louis* ]];
then 
    debug_config=" --config=config/debug.yaml"
else
    set -x
    export PATH="~/miniconda3/bin:$PATH"
    eval "$(conda shell.bash hook)"
    conda activate hybrid_wssd
fi


models=(bilstm_strong_100 bilstm_strong_100_dnn_wpe wet dry bilstm_unsupervised nara_wpe bilstm_strong bilstm_unsupervised_dnn_wpe bilstm_strong_dnn_wpe)
model="${models[$SLURM_ARRAY_TASK_ID]}"

# check if srun is available
if command -v srun 2>&1 >/dev/null
then 
  srun_if_available="srun "
else
  srun_if_available=""
fi
$srun_if_available python test_asr.py --model=$model
