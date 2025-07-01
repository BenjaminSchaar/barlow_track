#!/usr/bin/env bash

#SBATCH --job-name=barlow_training
#SBATCH --time=1-00:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

pwd; hostname; date

# Get all user flags
while getopts p:n: flag
do
    case "${flag}" in
        p) network_args_path=${OPTARG};;
        n) is_dry_run=${OPTARG};;
        *) raise error "Unknown flag"
    esac
done

BASE_CODE_PATH="/lisc/user/fieseler/zimmer/wbfm/code/barlow_track"
CMD=${BASE_CODE_PATH}/barlow_track/scripts/train_barlow_clusterer.py

echo "Running training on network yaml file: $network_args_path"
python $CMD -p $network_args_path
