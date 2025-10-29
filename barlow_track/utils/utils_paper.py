from wbfm.utils.general.utils_paper import apply_figure_settings
import numpy as np
import plotly
from scipy.spatial import cKDTree
from barlow_track.utils.utils_ground_truth import calculate_accuracy
from wbfm.utils.neuron_matching.utils_candidate_matches import rename_columns_using_matching
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px


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

    num_frames = project_data_barlow.num_frames_minus_tracking_failures()

    df_correlation = pd.DataFrame({
        'column': list(num_detections.keys()),
        'num_detections': list(num_detections.values()),
        'num_mismatches': list(num_mismatches.values()),
        'num_misses': list(num_misses.values()),
        'Number of gaps': list(num_nan.values()),
        'Fraction of gaps': np.array(list(num_nan.values())) / num_frames
    })
    df_correlation['Percent tracked'] = 100*(1 - df_correlation['Fraction of gaps'])
    df_correlation['Number of mistakes'] = df_correlation['num_mismatches'] + df_correlation['num_misses']
    df_correlation['Accuracy'] = 1 - df_correlation['Number of mistakes'] / df_correlation['num_detections']
    df_correlation['Fraction of mistakes'] = df_correlation['Number of mistakes'] / num_frames
    if use_traces:
        df_correlation['correlation'] = list(correlations.values())

    return df_correlation


def calculate_track_metrics_no_gt(project_data):
    gt_column='raw_segmentation_id'
    num_frames = project_data.num_frames_minus_tracking_failures()

    df = project_data.final_tracks.copy()
    col_gt = df.copy().loc[:, (slice(None), gt_column)].droplevel(1, axis=1)
    df_correlation = pd.DataFrame(col_gt.shape[0] - col_gt.count(), columns=['Number of gaps'])
    df_correlation['Fraction of gaps'] = df_correlation['Number of gaps'] / num_frames
    df_correlation['Percent tracked'] = 100*(1 - df_correlation['Fraction of gaps'])
    return df_correlation

def plot_hist_and_accuracy(df, dataset, max_bin=100, DEBUG=False):
    color = paper_colormap()[dataset]
    epsilon = 1e-6

    proxy_col = "Percent tracked"
    accuracy_col = "Accuracy"
    x_vals = df[proxy_col]
    # Log bins
    log_bins = np.logspace(np.log10(90), np.log10(max_bin), 10)
    log_bins[-1] += epsilon
    counts, edges = np.histogram(x_vals, bins=log_bins)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    bin_widths = edges[1:] - edges[:-1]

    # --- Histogram (left y-axis) ---
    hist_trace = go.Bar(
        x=bin_centers,
        y=100*counts / counts.sum(),
        width=bin_widths,
        name="Histogram (Fraction)",
        marker_color="lightgray",
        # opacity=0.5,
        yaxis="y"   # left y-axis
    )

    # --- Scatter (right y-axis) ---
    if accuracy_col in df:
        scatter_trace = go.Scatter(
            x=df[proxy_col],
            y=df[accuracy_col],
            mode="markers",
            name="Accuracy",
            marker=dict(color=color, size=3, opacity=0.5),
            yaxis="y2"  # right y-axis
        )

        # --- Trendline fit ---
        # We'll fit in log-space if x is log-distributed
        x_vals = df[proxy_col]
        X = x_vals
        y = df[accuracy_col]

        slope, intercept = np.polyfit(X, y, 1)
        y_pred = slope * X + intercept

        # Compute R²
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / ss_tot

        # Generate trendline points for plotting (sorted)
        x_sorted = np.sort(x_vals)
        y_trend = slope * x_sorted + intercept

        trend_trace = go.Scatter(
            x=x_sorted,
            y=y_trend,
            mode="lines",
            name=f"Trendline (slope={slope:.3f}, R²={r2:.3f})",
            line=dict(color=color, width=2),
            yaxis="y2"
        )

        # --- Combine traces ---
        fig = go.Figure(data=[hist_trace, scatter_trace, trend_trace])
    else:
        fig = go.Figure(data=hist_trace)

    # --- Layout ---
    fig.update_layout(
        xaxis=dict(
            title="Track Coverage (%)",
            type="log",
            range=[np.log10(log_bins[0]), np.log10(log_bins[-1])]
        ),
        yaxis=dict(
            title="Tracks (%)",
            side="left",
            showgrid=False,
            titlefont=dict(color="gray")
        ),
        # annotations=[
        #     dict(
        #         x=0.05,
        #         y=0.95,
        #         xref="paper",
        #         yref="paper",
        #         text=f"Slope: {slope:.2f}<br>R²: {r2:.2f}",
        #         showarrow=False,
        #         align="left",
        #         bgcolor="white",
        #         bordercolor="gray",
        #         borderwidth=1,
        #         opacity=0.8,
        #         font=dict(size=12)
        #     )
        # ],
        bargap=0.05,
        showlegend=False
    )
    if accuracy_col in df:
        fig.update_layout(
            yaxis2=dict(
                title="Accuracy",
                overlaying="y",
                side="right",
                showgrid=False,
                tickformat=".0%",
                color=color,
                range=[0.9, 1]
            ),
        )
    apply_figure_settings(fig, width_factor=0.3, height_factor=0.15)

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

