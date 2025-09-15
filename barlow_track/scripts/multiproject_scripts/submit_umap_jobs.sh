#!/bin/bash

ZARR_PATH="/lisc/scratch/neurobiology/zimmer/schwartz/traces_harvard_flip/2025_07_30trial_8/3-tracking/barlow_tracker/embedding.zarr"
OUT_DIR="/lisc/scratch/neurobiology/zimmer/schwartz/traces_harvard_flip/2025_07_30trial_8/"

for n_neighbors in 5 15 30
do
  for min_dist in 0.0 0.1 0.5
  do
    for metric in euclidean cosine
    do
      sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=umap_${n_neighbors}_${min_dist}_${metric}
#SBATCH --output=logs/umap_${n_neighbors}_${min_dist}_${metric}.out
#SBATCH --error=logs/umap_${n_neighbors}_${min_dist}_${metric}.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

module load python/3.10  # adjust to match your environment
source ~/venvs/umap/bin/activate  # or your env path

python umap_plot.py \
  --zarr_path $ZARR_PATH \
  --out_dir $OUT_DIR \
  --n_neighbors $n_neighbors \
  --min_dist $min_dist \
  --metric $metric
EOF
    done
  done
done
