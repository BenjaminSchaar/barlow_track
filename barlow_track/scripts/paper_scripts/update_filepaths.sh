#!/bin/bash

set -euo pipefail

# --- Configuration ---
old_str="/lisc/scratch/neurobiology/zimmer/fieseler"
new_str="/lisc/data/scratch/neurobiology/zimmer/fieseler"
file_name="project_config.yaml"

dry_run=false

# --- Parse arguments ---
for arg in "$@"; do
    case $arg in
        --dry-run)
            dry_run=true
            shift
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--dry-run]"
            exit 1
            ;;
    esac
done

echo "Searching for files named '$file_name'..."
echo "Replacing:"
echo "  OLD: $old_str"
echo "  NEW: $new_str"
echo "Dry-run mode: $dry_run"
echo

# --- Main logic ---
if $dry_run; then
    echo "Files that would be changed:"
    grep -rl --include="$file_name" "$old_str" . || echo "No matches found."
else
    echo "Performing replacements..."
    grep -rl --include="$file_name" "$old_str" . \
        | while read -r file; do
            echo "Updating: $file"
            sed -i "s|$old_str|$new_str|g" "$file"
        done
    echo "Done!"
fi