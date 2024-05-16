#!/bin/bash

# This script runs all the scripts in a folder (hardcoded)

target_dir="/home/charles/Current_work/barlow_networks/iclr_initial_projects"

# Use glob to list over subfolders
for folder in $target_dir/*; do
    # Check if folder is a directory
    if [ -d "$folder" ]; then
        # Check if folder contains the proper yaml file
        if [ -f "$folder/train_config.yaml" ]; then
            # Train the network
            echo "Training network defined in $folder"
            python ./train_barlow_clusterer.py -p "$folder/train_config.yaml"
        fi
    fi
done