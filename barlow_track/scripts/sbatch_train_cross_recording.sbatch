#!/usr/bin/env bash
#SBATCH --job-name=barlow_cross_rec
#SBATCH --time=3-00:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a30|v100"
#SBATCH --output=barlow_cross_rec_%j.out
#SBATCH --error=barlow_cross_rec_%j.err

pwd; hostname; date

while getopts p: flag
do
    case "${flag}" in
        p) network_args_path=${OPTARG};;
        *) echo "Unknown flag" && exit 1
    esac
done

if [ -z "$network_args_path" ]; then
    echo "Usage: sbatch sbatch_train_cross_recording.sbatch -p /path/to/config.yaml"
    exit 1
fi

BASE_CODE_PATH="/lisc/scratch/neurobiology/zimmer/wbfm/code/barlow_track"
CMD=${BASE_CODE_PATH}/barlow_track/scripts/train_barlow_cross_recording.py

echo "Config: $network_args_path"
echo "Script: $CMD"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python $CMD -p $network_args_path

echo "Done: $(date)"
