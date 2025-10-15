#!/bin/bash

base_dir="."

# Loop over directories only
for dir in "$base_dir"/*/; do
    # Remove trailing slash for convenience
    dir="${dir%/}"
    
    echo "Processing folder: $dir"
    
    cd "$dir"
    bash RUNME.sh -s alt_barlow_embedding -c
    cd ..
done
