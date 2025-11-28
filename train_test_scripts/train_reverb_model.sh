#!/bin/bash

#SBATCH --array=0-11
#SBATCH --partition=P100
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

cartesian_product=({ears,synthethic}+{null,rereverb}+{full,0.05,100})
current_config="${cartesian_product[$SLURM_ARRAY_TASK_ID]}"

OLDIFS=$IFS
IFS="+"
set -- $current_config
IFS=$OLDIFS

dataset=$1
loss=$2
fraction=$3
logger_name="$dataset"_"$loss"_"$fraction"

rir_model="$dataset""_blind_polack"
if [[ $loss =~ "null" ]]
then
    joint_loss="null"
    speech_model="null"
    trainer_ckpt_monitor="validation_step_DRR_MSE"
else
    joint_loss="linear_and_log_gradnorm"
    speech_model="oracle"
    trainer_ckpt_monitor="validation_step_rereverberation_loss_loss"
fi

if [[ "$fraction" == "full" ]]
then
  data_fraction_option=""
  trainer_max_epochs=100
else
  data_fraction_option=" --data.limit_training_size=$fraction"
  trainer_max_epochs=400
fi

echo `printf '_%.0s' {1..20}`
echo "dataset: $dataset"
echo "loss: $loss"
echo "fraction of the dataset: $fraction"
echo "logger name: $logger_name"
echo `printf '_%.0s' {1..20}`

# check if srun is available
if command -v srun 2>&1 >/dev/null
then 
  srun_if_available="srun "
else
  srun_if_available=""
fi

$srun_if_available python cli.py fit --config=config/trainer_and_optimizer.yaml --trainer.logger.init_args.name=taslp/reverb_model/"$logger_name" --data=config/data/"$dataset".yaml --config=config/speech_models/"$speech_model".yaml --config=config/rir_models/"$rir_model".yaml $data_fraction_option --data.enable_caching_train=false --config=config/joint_losses/"$joint_loss".yaml --trainer.max_epochs=$trainer_max_epochs --trainer.callbacks.init_args.monitor="$trainer_ckpt_monitor" --trainer.callbacks.init_args.mode="min" $further_options

# wait
