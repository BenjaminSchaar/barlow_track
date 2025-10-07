import logging
import os

import matplotlib
matplotlib.use("Agg")  # For headless opencv
import numpy as np
from matplotlib import pyplot as plt, patheffects as PathEffects
from matplotlib.colors import TwoSlopeNorm
from wbfm.utils.neuron_matching.utils_candidate_matches import rename_columns_using_matching
from wbfm.utils.performance.comparing_ground_truth import calculate_accuracy_from_dataframes


def visualize_model_performance(c, save_fname=None, vmin=None, vmax=None):
    """
    Plots the correlation matrix of the model's output, which should be close to diagonal (if trained)

    Parameters
    ----------
    c

    Returns
    -------

    """
    all_vals = c.cpu().numpy()

    if vmin is None:
        vmin = np.clip(np.nanmin(all_vals), -1, 0)
    if vmax is None:
        vmax = np.clip(np.nanmax(all_vals), 0, 1)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    fig = plt.figure()
    plt.imshow(all_vals[:10, :10], norm=norm, cmap='PiYG')
    plt.colorbar()
    # plt.show()
    if save_fname is not None:
        plt.savefig(save_fname)

    return fig


def plot_clusters(db, Y, class_labels=True, class_label_for_noise=False):
    """
    Plots the clusters found by a clustering algorithm (e.g., DBSCAN) in 2D space.

    Parameters
    ----------
    db : DBSCAN or similar clustering object
        The clustering object containing labels and core sample indices.
    Y : np.ndarray
        The 2D coordinates of the data points to be plotted. E.x. output of UMAP
    class_labels : bool, optional
        Whether to annotate the clusters with their labels (text placed at the cluster centroid). Default is True.
    class_label_for_noise : bool, optional
        Whether to annotate the noise points (-1 label) with a label. Default is False.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot of the clusters.
    """
    fig = plt.figure(figsize=(10, 10), dpi=300)

    if Y.shape[1] > 2:
        logging.warning("Data passed was not 2 dimensional (did you mean to run tsne?). For now, taking top 2")
        Y = Y[:, :2]

    if isinstance(db, np.ndarray):
        labels = db
    else:
        labels = db.labels_
    # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    # colors = [plt.cm.Set1(each) for each in np.linspace(0, 1, len(unique_labels))]
    colors = matplotlib.colors.ListedColormap(np.random.rand(256, 3)).colors
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k
        xy = Y[class_member_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        if class_labels:
            if k == -1 and not class_label_for_noise:
                continue
            text = plt.annotate(f'{k}', np.mean(xy, axis=0), fontsize=32, color='black')
            text.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

    plt.title("Estimated number of clusters: %d" % n_clusters_)
    plt.tight_layout()
    plt.show()

    return fig


def plot_relative_accuracy(df_combined, project_data, results_subfolder=None, to_save=True):
    df_base = project_data.get_final_tracks_only_finished_neurons()[0]
    if df_base is None or df_base.empty:
        project_data.logger.warning("No ground truth to compare to, using all tracks instead")
        if project_data.final_tracks is None:
            project_data.logger.warning("No tracks to compare to, skipping")
            return
        # Sometimes there may be a type mismatch between indices (float vs int)
        num_frames = df_combined.shape[0]
        df_base = project_data.final_tracks.iloc[:num_frames]
        
    df_cluster_renamed, matches, conf, name_mapping = rename_columns_using_matching(df_base, df_combined,
                                                                                    try_to_fix_inf=True)
    df_all_acc = calculate_accuracy_from_dataframes(df_base, df_cluster_renamed,
                                                    column_names=['raw_neuron_ind_in_list'])
    df_tracker = project_data.intermediate_global_tracks
    df_all_acc_original = calculate_accuracy_from_dataframes(df_base, df_tracker,
                                                             column_names=['raw_neuron_ind_in_list'])
    plt.figure(figsize=(20, 5), dpi=300)
    plt.xticks(rotation=90)
    plt.ylabel("Fraction correct (exc. gt nan)")
    plt.xlabel("Neuron name")
    plt.plot(df_all_acc_original.index, df_all_acc_original['matches_to_gt_nonnan'], label='Old tracker')
    plt.plot(df_all_acc.index, df_all_acc['matches_to_gt_nonnan'], '-o', label='Unsupervised tracker')
    plt.title(f"Tracking accuracy (mean={np.mean(df_all_acc['matches_to_gt_nonnan'])}")
    plt.legend()
    plt.tight_layout()

    if to_save:
        fname = os.path.join(results_subfolder, f'accuracy.png')
        fname = project_data.project_config.resolve_relative_path(fname)
        plt.savefig(fname)
