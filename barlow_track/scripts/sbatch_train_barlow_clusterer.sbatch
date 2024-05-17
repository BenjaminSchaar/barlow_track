#!/usr/bin/env bash

#SBATCH --job-name=barlow_training
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --time=8:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu,basic

pwd; hostname; date

# Get all user flags
while getopts t:n:s: flag
do
    case "${flag}" in
        t) project_path=${OPTARG};;
        n) is_dry_run=${OPTARG};;
        *) raise error "Unknown flag"
    esac
done

BASE_CODE_PATH="/lisc/user/fieseler/zimmer/wbfm/code/barlow_track"
CMD=${BASE_CODE_PATH}/barlow_track/scripts/train_barlow_clusterer.py

echo "Running training on project"
echo $project_path
python $CMD -p $project_path
