#!/bin/bash

# This script tracks a worm project using all trained barlow networks in a folder of networks (hardcoded)

# Get all user flags
while getopts t:n: flag
do
    case "${flag}" in
        t) project_path=${OPTARG};;
        n) is_dry_run=${OPTARG};;
        *) raise error "Unknown flag"
    esac
done

# Define hardcoded paths
network_folder="/home/charles/Current_work/barlow_networks/iclr_initial_projects"
tracking_cmd="/home/charles/Current_work/barlow_networks/wbfm/scripts/postprocessing/4+track_using_barlow.py"

# Loop through all networks in the folder
for network in $network_folder/*; do
    # Check if network is a directory
    if [ -d "$network" ]; then
        # Check if network contains a trained network
        if [ -f "$network/resnet50.pth" ]; then
          # TODO
            # Check if the project is already tracked
            path_to_tracking_results="$project_path/tracking_results.csv"
            if [ -f "$project_path/tracking_results.csv" ]; then
                echo "Project $project_path is already tracked"
                continue
            fi
            # Track
            echo "Tracking project $project_path using network defined in $network"
            if [ "$is_dry_run" ]; then
                echo "Dry run with command: python $tracking_cmd with project_path=$project_path model_fname=$network/resnet50.pth"
            else
                python "$tracking_cmd" with project_path="$project_path" model_fname="$network"/resnet50.pth
            fi
        fi
    fi
done