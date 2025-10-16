#!/bin/bash

old_str="/lisc/scratch/neurobiology/zimmer/fieseler/wbfm_projects_future/"
new_str="/lisc/data/scratch/neurobiology/zimmer/fieseler/barlow_track_paper/"
file_name="project_config.yaml"

find . -type f -name file_name -exec sed -i 's|$old_str|$new_str|g' {} +
