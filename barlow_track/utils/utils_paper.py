# Diff the multiindex dataframes, and calculate the euclidean zxy distance distribution
import numpy as np
import plotly
from scipy.spatial import cKDTree


def calculate_distance_diff(df):
    df = df.copy()
    dx = df.loc[:, (slice(None), 'x')].diff().values
    dy = df.loc[:, (slice(None), 'y')].diff().values
    dz = df.loc[:, (slice(None), 'z')].diff().values
    all_dxyz = np.stack([dx, dy, dz], axis=-1)
    distance_distribution = np.sqrt(np.sum(all_dxyz**2, axis=-1))
    return distance_distribution, all_dxyz


def calculate_nearest_neighbor_distance(df, return_all=False):
    # Extract x, y, z positions for all neurons at each time point
    x = df.loc[:, (slice(None), 'x')].values
    y = df.loc[:, (slice(None), 'y')].values
    z = df.loc[:, (slice(None), 'z')].values
    coords = np.stack([x, y, z], axis=2)  # shape: (frames, neurons, 3)

    mean_distance = []
    all_distances = np.zeros(coords.shape[:-1])
    all_distances[:] = np.nan  # Initialize with NaNs
    for t, frame_coords in enumerate(coords):
        # Remove neurons with any NaN coordinate in this frame
        valid = ~np.isnan(frame_coords).any(axis=1)
        points = frame_coords[valid]
        if len(points) < 2:
            mean_distance.append(np.nan)
            continue
        # Use cKDTree to find nearest neighbor distances
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=2)  # k=2 because the nearest is itself (distance 0)
        nn_dist = dists[:, 1]  # Take the second column (nearest neighbor, not itself)
        if return_all:
            all_distances[t, valid] = nn_dist
        else:
            mean_distance.append(np.mean(nn_dist))
    if return_all:
        return all_distances
    else:
        return np.array(mean_distance)


BASE_COLORMAP = plotly.colors.qualitative.D3
def paper_colormap():
    return {'Zimmer': BASE_COLORMAP[0], 'Samuel': BASE_COLORMAP[2], 'Flavell': BASE_COLORMAP[1]}
