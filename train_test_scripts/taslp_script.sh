#!/bin/bash

# -----------------------------
# UNCOMMENT THE SBATCH COMMANDS
# -----------------------------

if ((BASH_VERSINFO < 4)); then exit 1; fi

further_options="$@"
echo $further_options
## Check if running locally
hostname=`hostname`
if [[ ${hostname,,} == *louis* ]];
then
	# We're running locally
	#
    for ID in {0..11}
    do
        break
#		SLURM_ARRAY_TASK_ID=$ID bash train_supervised.sh $further_options
	done
    for ID in {0..20}
    do
        break
#		SLURM_ARRAY_TASK_ID=$ID bash train_weak.sh $further_options
    done
    for ID in {0..11}
    do
        break
#       SLURM_ARRAY_TASK_ID=$ID bash train_reverb_model.sh $further_options
	done
    for ID in {0..11}
    do
        break
#        SLURM_ARRAY_TASK_ID=$ID bash train_unsupervised.sh $further_options
	done
else
	# We are running on GPU-GW or JZ

    # ------------------
    # Strong supervision
    # ------------------
	# Use the options to split on P100 and V100 machines
	# ears_bilstm_oraclerir on V100 to fit rir cache in ram
#	sbatch --array=1 --partition=V100 train_supervised.sh $further_options
	# BiLSTM on P100 or V100
#	sbatch --array=0,2-3 --partition=P100,V100 train_supervised.sh $further_options
	# FSN and tflocoformer on hiaudio,A100,L40S,A40,V100-32GB
#	sbatch --array=4-11 train_supervised.sh $further_options
    
    # ----------------
    # Weak supervision
    # ----------------
    # All on >16GB GPUS
#    sbatch --array=2-17:3 train_weak.sh $further_options
    # For reviewers @16kHz
#    sbatch --array=0,1,2,6,7,8,12,13,14%2 train_weak.sh $further_options

    # -----------------
    # Reverb model only
    # -----------------
#    sbatch --array=0,1,2,6,7,8 --partition=P100 train_reverb_model.sh $further_options
#    sbatch --array=3,4,5,9,10,11 --partition=A40 train_reverb_model.sh $further_options
    # For rewiewers redo exps on ears 16kHz
#    sbatch --array=0,1,2%1 --partition=P100 train_reverb_model.sh $further_options
#    sbatch --array=3,4,5%1 --partition=A40 train_reverb_model.sh $further_options


    # -------------------------
    # Unsupervised speech model
    # -------------------------
#    sbatch train_unsupervised.sh $further_options
fi
# bit of cleaning
# python -c "from model.utils.run_management import move_all_except_lastest_version_containing_checkpoint_to_trash; move_all_except_lastest_version_containing_checkpoint_to_trash(dry_run=False)"

# if command -v sbatch 2>&1 >/dev/null
# then 
#   begin="sbatch "
# else
#   begin=""
# fi
