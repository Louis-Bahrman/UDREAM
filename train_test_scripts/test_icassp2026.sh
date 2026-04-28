#!/bin/bash

dataset_name= "ears_reverb"

## Polack with scheduler
names=("ears16_FSN_monoband_compressed_linear" "ears16_FSN_monoband_linear" "ears16_FSN_monoband_logloss" "ears16_FSN_monoband_phase_inv_mse" "ears16_PhaseInvFSN_monoband_logloss")
for name in "${names[@]}"; do
    latest_model_dir="lightning_logs/ICASSP2026/"$name
    version=`ls --color=never $latest_model_dir | sort -V | tail -n 1`
    name_and_version=$name"/"$version
    logger_name_and_version="ICASSP2026/"$name_and_version #latest_exp/ears16_BiLSTM_bandwise10a100s_logloss/version_0"

    ckpt_name=`ls --color=never lightning_logs/"$logger_name_and_version"/checkpoints | sort -Vr | head -1`
    echo "using checkpoint" $ckpt_name
    logger_path="lightning_logs/"$logger_name_and_version"/checkpoints/"$ckpt_name

    dataset_params=" --data=config/data/ears.yaml --data.init_args.batch_size=1 --data.init_args.enable_caching_train=False --data.init_args.enable_caching_val=False --data.init_args.return_rir=False"
    config_file="lightning_logs/"$logger_name_and_version"/config.yaml"
    python cli.py test --config $config_file $dataset_params --model.speech_model_ckpt_path "$logger_path" --trainer.logger.init_args.name "tests/"$logger_name_and_version --model.reverb_model=null --model.joint_loss_module=null
done
# wait
