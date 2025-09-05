import argparse
from pathlib import Path
import zarr
import numpy as np
import umap
import matplotlib.pyplot as plt

def read_embedding_zarr(path):
    return zarr.open(str(path), mode='r')[:]

def plot_embedding(embedding_2d, out_path):
    plt.figure(figsize=(8, 8))
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=1)
    plt.title("UMAP projection")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()

def main(args):
    print("reading data ...")
    embedding = read_embedding_zarr(args.zarr_path)
    print("creating umap...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.seed
    )
    embedding_2d = reducer.fit_transform(embedding)
    print("plotting ...")
    out_file = Path(args.out_dir) / f"umap_n{args.n_neighbors}_d{args.min_dist}_m{args.metric}.png"
    plot_embedding(embedding_2d, out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--metric", type=str, default='euclidean')
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
