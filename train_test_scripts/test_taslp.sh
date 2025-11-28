#!/bin/bash

#SBATCH --partition=P100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --array=0-59%10

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

taslp_path="lightning_logs/taslp"

weak_logger_paths=($(ls -d $taslp_path/weak/*))
strong_logger_paths=($(ls -d $taslp_path/supervised/*))
reverb_model_logger_paths=($(ls -d $taslp_path/reverb_model/*))
unsupervised_logger_paths=($(ls -d $taslp_path/unsupervised/*))
taslp_marius_ablations_phase_and_strong_paths=($(ls -d $taslp_path/taslp_marius_ablations_phase_and_strong/*))

all_logger_paths=( "${strong_logger_paths[@]}" "${weak_logger_paths[@]}" "${reverb_model_logger_paths[@]}" "${unsupervised_logger_paths[@]}" "${taslp_marius_ablations_phase_and_strong_paths[@]}" )

logger_path=${all_logger_paths[$SLURM_ARRAY_TASK_ID]}
logger_name=`basename $logger_path`
supervision_type=`basename $(dirname ${logger_path})`


# config_file=`find $logger_path/*/config.yaml`
config_file=$(find $logger_path/ -type f -name "config.yaml" -print | xargs ls -t | head -n 1)
echo "config file: $config_file"

if [ -z "${config_file}" ];
then
    echo "No config file found, exiting"
    exit 1
fi

if [[ $logger_name == *synth* ]];
then
  dataset_names=( "synth" "ears" )
else
  dataset_names=( "ears" )
fi

# check if srun is available
if command -v srun 2>&1 >/dev/null
then
  srun_if_available="srun "
else
  srun_if_available=""
fi

for dataset_name in ${dataset_names[@]};
do
  if [[ $dataset_name == *synth* ]];
  then
    dataset_params=" --data.class_path=datasets.PairedDataModule --data.init_args.path=./data/test_ears_same"
  else
    dataset_params=" --data=config/data/ears.yaml --data.init_args.batch_size=1 --data.init_args.enable_caching_train=False --data.init_args.enable_caching_val=False --data.init_args.return_rir=True --data.init_args.resample_rir=True"
  fi
  if [[ $supervision_type == reverb_model ]]
  then
    # we need not to find the best checkpoint, but the latest.
    latest_ckpt_command="from model.utils.run_management import get_latest_checkpoint; print(get_latest_checkpoint('"$logger_path"'))"
    latest_ckpt_path=`python -c "$latest_ckpt_command"`
    $srun_if_available python cli.py test --config $config_file $dataset_params --model.reverb_model_ckpt_path "$latest_ckpt_path" --trainer.logger.init_args.name "test/$supervision_type/$logger_name/$dataset_name" --config=config/speech_models/oracle.yaml --config=config/joint_losses/linear_and_log.yaml
  else
    $srun_if_available python cli.py test --config $config_file $dataset_params --model.speech_model_ckpt_path "$logger_path" --trainer.logger.init_args.name "test/$supervision_type/$logger_name/$dataset_name" --model.reverb_model=null --model.joint_loss_module=null
  fi
done

wait
