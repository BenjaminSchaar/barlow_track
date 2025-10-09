#!/bin/bash

# This script updates the config files of multiple projects to ensure they are consistent with the folder name
CMD="/lisc/data/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm/scripts/folder_analysis/copy_missing_config_file_to_multiple_projects.sh"
top_level_project_folder="/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/analyzed_projects"

# Helper function that loops through the parent and updates each subfolder; the file structure is like:
# top_level_project_folder/name1/subfolder
# top_level_project_folder/name2/subfolder
# ...
update_configs_for_all_labs() {
    local parent_folder="$1"
    local subfolder="$2"
    local cfg_to_update="$3"
    for name_folder in "$parent_folder"/*; do
        if [ -d "$name_folder" ]; then
            full_path="$name_folder/$subfolder"
            if [ -d "$full_path" ]; then
                echo "Updating config in $full_path"
                bash "$CMD" -h
                # bash "$CMD" -t "$full_path" -c "$cfg_to_update" -n
            else
                echo "Subfolder $subfolder does not exist in $name_folder, skipping."
            fi
        fi
    done
}


# Subfolders to update

# This folder shouldn't be run with the new tracker, so use the default non-barlow snakemake config
subfolder="baseline_old_tracklets"
cfg_to_update="/lisc/data/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm/new_project_defaults/snakemake/snakemake_config.yaml"

update_configs_for_all_labs "$top_level_project_folder" "$subfolder" "$cfg_to_update"
