#!/bin/bash

#SBATCH --array=0-11
#SBATCH --partition=hiaudio,A100,L40S
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --time=23:00:00
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

cartesian_product=(bilstm+{ears,synthethic}+{rereverb,null}+{"0.05","100",full})
current_config="${cartesian_product[$SLURM_ARRAY_TASK_ID]}"

OLDIFS=$IFS
IFS="+"
set -- $current_config
IFS=$OLDIFS

speech_model=$1
dataset=$2
reverb_model_training_target=$3
reverb_model_training_fraction=$4
logger_name="$dataset"_"$speech_model"_"$reverb_model_training_target"_"$reverb_model_training_fraction"

rir_model="$dataset""_blind_polack"
joint_loss="linear_and_log_gradnorm"

# Best reverb model is already selected because checkpointing has been done properly 
# So the default options of reverb_model.load_state_dict_from_joint_model should be fine

reverb_model_logger_name="$dataset"_"$reverb_model_training_target"_"$reverb_model_training_fraction"
reverb_model_logger_path="./lightning_logs/taslp/reverb_model/$reverb_model_logger_name"

echo `printf '_%.0s' {1..20}`
echo "dataset: $dataset"
echo "speech model: $speech_model"
echo "reverb model training target: $reverb_model_training_target"
echo "reverb model training fraction: $reverb_model_training_fraction"
echo "logger name: $logger_name"
echo "reverb model logger name: $rir_model_logger_name"
echo `printf '_%.0s' {1..20}`

# check if srun is available
if command -v srun 2>&1 >/dev/null
then 
  srun_if_available="srun "
else
  srun_if_available=""
fi

$srun_if_available python cli.py fit --config=config/trainer_and_optimizer.yaml --trainer.logger.init_args.name=taslp/unsupervised/"$logger_name" --data=config/data/"$dataset".yaml --config=config/speech_models/"$speech_model".yaml --config=config/rir_models/"$rir_model".yaml --config=config/joint_losses/"$joint_loss".yaml --model.train_speech_model=true --model.train_reverb_model=false --model.reverb_model_ckpt_path="$reverb_model_logger_path" $further_options

# wait
