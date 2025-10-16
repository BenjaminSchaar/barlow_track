from wbfm.utils.general.utils_paper import apply_figure_settings
import numpy as np
import plotly
from scipy.spatial import cKDTree
from barlow_track.utils.utils_ground_truth import calculate_accuracy
from wbfm.utils.neuron_matching.utils_candidate_matches import rename_columns_using_matching
import plotly.graph_objects as go
import pandas as pd


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


def calculate_track_metrics(project_data_gt, project_data_barlow, use_traces=False, gt_column='raw_segmentation_id'):
    gt_names = project_data_gt.finished_neuron_names()

    df_gt = project_data_gt.final_tracks.copy().iloc[:-1, :]
    if len(gt_names) > 0:
        df_gt = df_gt[gt_names].copy()
    df_barlow = project_data_barlow.final_tracks.copy()
    
    if use_traces:
        df_gt_traces = project_data_gt.calc_default_traces(interpolate_nan=True, min_nonnan=0, filter_mode=None).copy().iloc[:-1, :][gt_names]
        df_barlow_traces = project_data_barlow.calc_default_traces(interpolate_nan=True, min_nonnan=0, filter_mode=None).copy().iloc[:-1, :]
        df_barlow_traces_nointerp = project_data_barlow.calc_default_traces(interpolate_nan=False, min_nonnan=0, filter_mode=None).copy().iloc[:-1, :]
    
    
        df_barlow_traces_renamed = df_barlow_traces.rename(columns=name_mapping)[list(name_mapping.values())].drop('unmatched_neuron', axis=1)
        df_barlow_traces_nointerp = df_barlow_traces_nointerp.rename(columns=name_mapping)[list(name_mapping.values())].drop('unmatched_neuron', axis=1)
        df_barlow_traces.shape, df_barlow_traces_renamed.shape, df_gt_traces.shape, len(name_mapping)

    df_barlow_renamed, matches, conf, name_mapping = rename_columns_using_matching(df_gt, df_barlow, column=gt_column)

    col_gt = df_gt.copy().loc[:, (slice(None), gt_column)].droplevel(1, axis=1)
    col_barlow = df_barlow_renamed.copy().loc[:, (slice(None), gt_column)].droplevel(1, axis=1)

    results = calculate_accuracy(col_gt, col_barlow)

    correlations = {}
    num_misses = {}
    num_mismatches = {}
    num_nan = {}
    num_detections = {}
    for col in df_gt.columns.get_level_values(0).unique():
        if col in df_barlow_renamed.columns.get_level_values(0).unique():
            if use_traces:
                correlations[col] = df_gt_traces[col].corr(df_barlow_traces_renamed[col])
            num_misses[col] = results['misses'][col].sum()
            num_mismatches[col] = results['mismatches'][col].sum()
            num_nan[col] = df_barlow_renamed.shape[0] - df_barlow_renamed[col][gt_column].count()
            num_detections[col] = df_gt[col][gt_column].count()

    df_correlation = pd.DataFrame({
        'column': list(num_detections.keys()),
        'num_detections': list(num_detections.values()),
        'num_mismatches': list(num_mismatches.values()),
        'num_misses': list(num_misses.values()),
        'Number of gaps': list(num_nan.values()),
        'Fraction of gaps': np.array(list(num_nan.values())) / project_data_barlow.num_frames
    })
    df_correlation['Number of mistakes'] = df_correlation['num_mismatches'] + df_correlation['num_misses']
    df_correlation['Accuracy'] = 1 - df_correlation['Number of mistakes'] / df_correlation['num_detections']
    df_correlation['Fraction of mistakes'] = df_correlation['Number of mistakes'] / project_data_barlow.num_frames
    if use_traces:
        df_correlation['correlation'] = list(correlations.values())

    return df_correlation


def plot_hist_and_accuracy(df, dataset, max_bin=0.25):
    color = paper_colormap()[dataset]
    
    proxy_col = "Fraction of gaps"
    accuracy_col = "Accuracy"
    
    # Small epsilon to avoid log(0)
    epsilon = 1e-6
    x_vals = df[proxy_col].clip(lower=epsilon)
    
    # Define log-spaced bins
    log_bins = np.logspace(np.log10(np.min([5e-3, x_vals.min()])), 
                        #    np.log10(np.max([max_bin, x_vals.max()])), 
                           np.log10(np.max([max_bin])), 
                           10)
    
    # Make sure the last bin is larger than the max value
    log_bins[-1] += epsilon
    
    # Bin counts (for histogram)
    counts, edges = np.histogram(x_vals, bins=log_bins)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    bin_widths = edges[1:] - edges[:-1]
    
    # Histogram trace
    hist_trace = go.Bar(
        x=bin_centers,
        y=counts / counts.sum(),
        width=bin_widths,
        name=proxy_col,
        marker_color="lightgray",
        # opacity=0.9,
        yaxis="y1"
    )
    
    # Create figure
    fig = go.Figure()
    fig.add_trace(hist_trace)
    
    if accuracy_col in df.columns:
        for i, interval in enumerate(log_bins[:-1]):
            bin_data = df[(df[proxy_col] >= log_bins[i]) & (df[proxy_col] < log_bins[i+1])][accuracy_col]
            if len(bin_data) == 0:
                continue  # skip empty bins
            fig.add_trace(
                go.Box(
                    y=bin_data,#[accuracy_col],
                    x=[bin_centers[i]] * len(bin_data),  # position at bin center
                    marker_color=color,
                    boxpoints=False,#"outliers",  # show individual points
                    # width=bin_widths[i],
                    yaxis="y2",
                    name="Accuracy",
                    showlegend=(i==0),
                )
            )
    
    # Layout
    fig.update_layout(
        xaxis=dict(title=f"log({proxy_col})", type="log"),
        yaxis=dict(title="Tracks<br>(Fraction)", rangemode="tozero", showgrid=False, titlefont=dict(color='gray')),
        yaxis2=dict(title="Accuracy", overlaying="y", side="right", #range=[0.75,1], 
                    showgrid=False, tickformat=".0%", 
                    color=color#titlefont=dict(color=color), #type='log'
                   ),
        bargap=0.05,
        legend=dict(x=1.15, y=0.9),
        showlegend=False,
    )
    
    apply_figure_settings(fig, width_factor=0.35, height_factor=0.15)
    
    fig.show()

    return fig


BASE_COLORMAP = plotly.colors.qualitative.D3
def paper_colormap():
    _cmap = {'Zimmer': BASE_COLORMAP[0], 'Leifer': BASE_COLORMAP[1],# 'Samuel': BASE_COLORMAP[2], 
             'Flavell': BASE_COLORMAP[2]}
    cmap = _cmap.copy()
    for k, v in _cmap.items():
        cmap[k.lower()] = v
    return cmap


def paper_category_order():
    _cmap = {'Dataset': ['Zimmer', 'Leifer', 'Flavell', 'zimmer', 'leifer', 'flavell']}
    cmap = _cmap.copy()
    for k, v in _cmap.items():
        cmap[k.lower()] = v
    return cmap

