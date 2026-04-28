#!/bin/bash

further_options="$@"
for ID in $(seq 0 4);
do
    case $ID in
    # ############# Experiments phase invariant loss vs phase dependant loss & compressed vs raw #############
        0)
            logger_name="ears16_FSN_monoband_logloss"
            config_tail="--data=config/data/ears.yaml --config=config/speech_models/fsn.yaml --config=config/rir_models/nonblind_polack.yaml --config=config/rir_models/ears_rir16kHz.yaml --config=config/joint_losses/log.yaml"
            special_configs="--optimizer.init_args.lr=1e-4"
            ;;
        1)
            logger_name="ears16_FSN_monoband_linear"
            config_tail="--data=config/data/ears.yaml --config=config/speech_models/fsn.yaml --config=config/rir_models/nonblind_polack.yaml --config=config/rir_models/ears_rir16kHz.yaml --config=config/joint_losses/linear.yaml"
            special_configs="--optimizer.init_args.lr=1e-4"
            ;;
        2)
            logger_name="ears16_FSN_monoband_compressed_linear"
            config_tail="--data=config/data/ears.yaml --config=config/speech_models/fsn.yaml --config=config/rir_models/nonblind_polack.yaml --config=config/rir_models/ears_rir16kHz.yaml --config=config/joint_losses/linear_compressed.yaml"
            special_configs="--optimizer.init_args.lr=1e-4"
            ;;
        3)
            logger_name="ears16_FSN_monoband_phase_inv_mse"
            config_tail="--data=config/data/ears.yaml --config=config/speech_models/fsn.yaml --config=config/rir_models/nonblind_polack.yaml --config=config/rir_models/ears_rir16kHz.yaml --config=config/joint_losses/phase_invariant_mse.yaml"
            special_configs="--optimizer.init_args.lr=1e-4"
            ;;
    # ############ Phase invariant FSN with phase invariant loss ################
        4)
            logger_name="ears16_PhaseInvFSN_monoband_logloss"
            config_tail="--data=config/data/ears.yaml --config=config/speech_models/phase_invariant_fsn.yaml --config=config/rir_models/nonblind_polack.yaml --config=config/rir_models/ears_rir16kHz.yaml --config=config/joint_losses/log.yaml"
            special_configs="--optimizer.init_args.lr=1e-4"
            ;;
    esac
    echo "Running experiment: $logger_name"
    which python
    python cli.py fit --config=config/trainer_and_optimizer.yaml --trainer.logger.init_args.name="ICASSP2026/$logger_name" $config_tail $special_configs
done