#!/bin/bash

base_dir="."

# Loop over directories only
for dir in "$base_dir"/*/; do
    # Remove trailing slash for convenience
    dir="${dir%/}"
    
    echo "Processing folder: $dir"
    
    cd "$dir/snakemake"
    bash RUNME.sh -s traces -c
    cd ../..
done
