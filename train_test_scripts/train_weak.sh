#!/bin/bash

#SBATCH --array=0-17
#SBATCH --partition=hiaudio,A100,L40S,A40
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

cartesian_product=({bilstm,fsn,tflocoformer}+{ears,synthethic}+{1draw,10draws,10drawsbestchannel})
current_config="${cartesian_product[$SLURM_ARRAY_TASK_ID]}"

OLDIFS=$IFS
IFS="+"
set -- $current_config
IFS=$OLDIFS

dataset=$2
speech_model=$1
rir_variant=$3
logger_name="$dataset"_"$speech_model"_"$rir_variant"

rir_model="nonblind_polack"
if [[ "$rir_variant" =~ bestchannel ]]
then
    joint_loss="linear_and_log_gradnorm_bestchannel"
else
    joint_loss="linear_and_log_gradnorm"
fi

if [[ "$rir_variant" =~ 10draws ]]
then
  rir_further_options=" --config=config/rir_models/10_polack_draws.yaml --config=config/accumulate_grad.yaml"
else
  rir_further_options=""
fi

if [[ ${dataset,,} == synthethic ]]
then
  rir_further_options="$rir_further_options --config=config/rir_models/positive_valued.yaml --config=config/rir_models/synthethic.yaml"
else
  # Changed for review
  # rir_further_options="$rir_further_options --config=config/rir_models/ears_rir48kHz.yaml"
  rir_further_options="$rir_further_options --config=config/rir_models/ears_rir16kHz.yaml"
fi

echo `printf '_%.0s' {1..20}`
echo "dataset: $dataset"
echo "speech model: $speech_model"
echo "rir variant: $rir_variant"
echo "logger name: $logger_name"
echo `printf '_%.0s' {1..20}`

# check if srun is available
if command -v srun 2>&1 >/dev/null
then 
  srun_if_available="srun "
else
  srun_if_available=""
fi

$srun_if_available python cli.py fit --config=config/trainer_and_optimizer.yaml --trainer.logger.init_args.name=taslp/weak/"$logger_name" --data=config/data/"$dataset".yaml --config=config/speech_models/"$speech_model".yaml --config=config/rir_models/"$rir_model".yaml $rir_further_options --config=config/joint_losses/"$joint_loss".yaml $further_options

# wait
