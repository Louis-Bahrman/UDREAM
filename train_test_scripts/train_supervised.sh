#!/bin/bash

#SBATCH --array=0-11
#SBATCH --partition=hiaudio,A100,L40S,A40,V100-32GB
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
# #SBATCH --time=04:00:00
#SBATCH --signal=SIGUSR1@90

## Check if running locally
hostname=`hostname`
if [[ ${hostname,,} == *louis* ]];
then
	debug_config=" --config=config/debug.yaml"
#	debug_config=" --print_config"
else
	set -x
	export PATH="~/miniconda3/bin:$PATH"
	eval "$(conda shell.bash hook)"
	conda activate hybrid_wssd
	debug_config=""
fi
 
further_options="$debug_config $@"

cartesian_product=({bilstm,fsn,tflocoformer}+{ears,synthethic}+{null,oraclerir})
current_config="${cartesian_product[$SLURM_ARRAY_TASK_ID]}"

OLDIFS=$IFS
IFS="+"
set -- $current_config
IFS=$OLDIFS

dataset=$2
speech_model=$1
rir_model=$3
logger_name="$dataset"_"$speech_model"_"$rir_model"

if [[ $rir_model != null ]]
then
	rir_model="$dataset"_"$rir_model"
	joint_loss="linear_and_log_gradnorm"
else
	joint_loss="null"
fi

echo `printf '_%.0s' {1..20}`
echo "dataset: $dataset"
echo "speech model: $speech_model"
echo "rir model: $rir_model"
echo "logger name: $logger_name"
echo `printf '_%.0s' {1..20}`

# check if srun is available
if command -v srun 2>&1 >/dev/null
then 
  srun_if_available="srun "
else
  srun_if_available=""
fi

$srun_if_available python cli.py fit --config=config/trainer_and_optimizer.yaml --trainer.logger.init_args.name=taslp/supervised/"$logger_name" --data=config/data/"$dataset".yaml --config=config/speech_models/"$speech_model".yaml --config=config/rir_models/"$rir_model".yaml --config=config/joint_losses/"$joint_loss".yaml $further_options

# wait
