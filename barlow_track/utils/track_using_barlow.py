import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import zarr
from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD
from tqdm.auto import tqdm
from wbfm.utils.neuron_matching.utils_candidate_matches import rename_columns_using_matching
from wbfm.utils.performance.comparing_ground_truth import calculate_accuracy_from_dataframes
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.projects.utils_filenames import pickle_load_binary
from wbfm.utils.projects.utils_neuron_names import name2int_neuron_and_tracklet
from wbfm.barlow_project.utils.track_using_clusters import WormTsneTracker


def track_using_barlow_from_config(project_config: ModularProjectConfig,
                                   model_fname=None,
                                   results_subfolder=None,
                                   to_plot_relative_accuracy=True):
    """
    Tracks a project using a pretrained Barlow Twins model

    Can reuse prior steps if they have already been run. Runs in this order:
    1. Load the pretrained neural network
    2. Embed the data (volumetric images) using the neural network
    3. Build a class to organize the embeddings and the clusterer
    4. Run the clusterer and get the final tracks
    5. Calculate accuracy and save the results

    Parameters
    ----------
    project_config
    model_fname
    results_subfolder
    to_plot_relative_accuracy

    Returns
    -------

    """
    if model_fname is None:
        model_fname = 'checkpoint_barlow_small_projector'
    if results_subfolder is None:
        # The default folder is built from the model fname, but removes the "checkpoint_" prefix
        # Also if it is a full path, just take the last foldername
        if Path(model_fname).is_absolute():
            results_subfolder = os.path.split(model_fname)[-2]
        else:
            results_subfolder = model_fname
        results_subfolder = results_subfolder.replace('checkpoint_', '')
        results_subfolder = f'3-tracking/{results_subfolder}'

    if not isinstance(project_config, ModularProjectConfig):
        project_config = ModularProjectConfig(str(project_config))
    project_data = ProjectData.load_final_project_data_from_config(project_config)

    # Check to see if the results already exist
    results_subfolder_full = project_config.resolve_relative_path(results_subfolder)

    tracker_fname = os.path.join(results_subfolder_full, 'worm_tracker_barlow.pickle')
    if Path(tracker_fname).exists():
        project_config.logger.info("Found already saved tracker, loading...")
        tracker = pickle_load_binary(tracker_fname)

    else:
        # Next try: load metadata
        embedding_fname = os.path.join(results_subfolder_full, 'embedding.zarr')
        if Path(embedding_fname).exists():
            project_config.logger.info("Found already saved embedding files, loading...")
            X = np.array(zarr.open(embedding_fname))

            fname = os.path.join(results_subfolder_full, 'time_index_to_linear_feature_indices.pickle')
            time_index_to_linear_feature_indices = pickle_load_binary(fname)
            fname = os.path.join(results_subfolder_full, 'linear_ind_to_raw_neuron_ind.pickle')
            linear_ind_to_raw_neuron_ind = pickle_load_binary(fname)

            opt = dict(time_index_to_linear_feature_indices=time_index_to_linear_feature_indices,
                       svd_components=50,
                       cluster_directly_on_svd_space=True,
                       n_clusters_per_window=3,
                       n_volumes_per_window=120,
                       linear_ind_to_raw_neuron_ind=linear_ind_to_raw_neuron_ind)
            tracker = WormTsneTracker(X, **opt)
        else:
            tracker = None

    #
    if tracker is None:
        # Initialize a pretrained model
        # See: barlow_twins_evaluate_scratch
        if Path(model_fname).is_absolute():
            fname = model_fname
            project_config.logger.info(f"Using pretrained neural network: {fname}")
        else:
            # My draft networks are here
            project_config.logger.warning("Using draft networks; if you want to use the final networks, use an absolute path")
            folder_fname = '/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/nn_ideas/'
            fname = os.path.join(folder_fname, model_fname, 'resnet50.pth')

        gpu, model, target_sz = load_barlow_model(fname)
        model.eval()

        # Embed using the model
        all_embeddings = embed_using_barlow(gpu, model, project_data, target_sz)

        linear_ind_to_gt_ind, linear_ind_to_raw_neuron_ind, time_index_to_linear_feature_indices = build_embedding_metadata(
            all_embeddings, project_data)

        svd_components = 50
        X = np.vstack([np.vstack(list(emb.values())) for emb in all_embeddings.values()])
        alg = TruncatedSVD(n_components=svd_components)
        X_svd = alg.fit_transform(X)

        # Save embeddings and trackers
        opt = dict(time_index_to_linear_feature_indices=time_index_to_linear_feature_indices,
                   svd_components=svd_components,
                   cluster_directly_on_svd_space=True,
                   n_clusters_per_window=3,
                   n_volumes_per_window=120,
                   linear_ind_to_raw_neuron_ind=linear_ind_to_raw_neuron_ind)
        tracker = WormTsneTracker(X_svd, **opt)
        tracker_no_svd = WormTsneTracker(X, **opt)

        save_intermediate_results(X, linear_ind_to_gt_ind, linear_ind_to_raw_neuron_ind, project_config, project_data,
                                  time_index_to_linear_feature_indices, tracker, tracker_no_svd,
                                  subfolder=results_subfolder)

    # Do the clustering
    df_combined = tracker.track_using_global_clusterer()

    fname = os.path.join(results_subfolder, f'df_barlow_tracks.h5')
    project_config.save_data_in_local_project(df_combined, fname, make_sequential_filename=True)

    if to_plot_relative_accuracy:
        plot_relative_accuracy(df_combined, results_subfolder, project_data)


def plot_relative_accuracy(df_combined, results_subfolder, project_data, to_save=True):
    num_frames = df_combined.shape[0] - 1
    df_base = project_data.get_final_tracks_only_finished_neurons()[0].loc[:num_frames, :]
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
    plt.plot(df_all_acc.index, df_all_acc['matches_to_gt_nonnan'], label='Unsupervised tracker')
    plt.title(f"Tracking accuracy (mean={np.mean(df_all_acc['matches_to_gt_nonnan'])}")
    plt.legend()
    plt.tight_layout()

    if to_save:
        fname = os.path.join(results_subfolder, f'accuracy.png')
        fname = project_data.project_config.resolve_relative_path(fname)
        plt.savefig(fname)


def embed_using_barlow(gpu, model, project_data, target_sz):
    project_data.project_config.logger.info("Embedding using Barlow model")
    from wbfm.barlow_project.utils.barlow import NeuronImageWithGTDataset
    num_frames = project_data.num_frames - 1
    dataset = NeuronImageWithGTDataset.load_from_project(project_data, num_frames, target_sz)
    names = dataset.which_neurons
    all_embeddings = defaultdict(dict)
    with torch.no_grad():
        for t, (batch, ids) in tqdm(enumerate(dataset), total=len(dataset)):
            for name in names:
                if name in ids:
                    idx = ids.index(name)

                    crop = torch.unsqueeze(batch[:, idx, ...], 0).to(gpu)
                    all_embeddings[name][t] = model.embed(crop).cpu().numpy()
    return all_embeddings


def load_barlow_model(model_fname):
    from wbfm.barlow_project.utils.barlow import BarlowTwins3d
    from wbfm.barlow_project.utils.siamese import ResidualEncoder3D
    state_dict = torch.load(model_fname)
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = pickle_load_binary(Path(model_fname).with_name('args.pickle'))
    target_sz = np.array([4, 128, 128])
    backbone_kwargs = dict(in_channels=1, num_levels=2, f_maps=4, crop_sz=target_sz)
    model = BarlowTwins3d(args, backbone=ResidualEncoder3D, **backbone_kwargs).to(gpu)
    model.load_state_dict(state_dict)
    return gpu, model, target_sz


def save_intermediate_results(X, linear_ind_to_gt_ind, linear_ind_to_raw_neuron_ind, project_config, project_data,
                              time_index_to_linear_feature_indices, tracker, tracker_no_svd,
                              subfolder):
    fname = f'{subfolder}/worm_tracker_barlow.pickle'
    project_config.pickle_data_in_local_project(tracker, fname)
    fname = f'{subfolder}/worm_tracker_barlow_full.pickle'
    project_config.pickle_data_in_local_project(tracker_no_svd, fname)
    fname = f'{subfolder}/embedding.zarr'
    fname = project_data.project_config.resolve_relative_path(fname)
    z = zarr.open_array(fname, shape=X.shape, chunks=(10000, 256))
    z[:] = X
    fname = f'{subfolder}/time_index_to_linear_feature_indices.pickle'
    project_data.project_config.pickle_data_in_local_project(time_index_to_linear_feature_indices, fname)
    fname = f'{subfolder}/linear_ind_to_raw_neuron_ind.pickle'
    project_data.project_config.pickle_data_in_local_project(linear_ind_to_raw_neuron_ind, fname)
    fname = f'{subfolder}/linear_ind_to_gt_ind.pickle'
    project_data.project_config.pickle_data_in_local_project(linear_ind_to_gt_ind, fname)


def build_embedding_metadata(all_embeddings, project_data):
    # Collect metadata
    df_tracks = project_data.get_final_tracks_only_finished_neurons()[0]
    X = []
    time_index_to_linear_feature_indices = defaultdict(list)
    linear_ind_to_raw_neuron_ind = {}
    linear_ind_to_gt_ind = {}
    i_linear_ind = 0
    for name, vols_all_times in tqdm(all_embeddings.items()):
        t_list = list(vols_all_times.keys())
        vols_array = np.vstack(list(vols_all_times.values()))

        df_this_neuron = df_tracks[name, 'raw_neuron_ind_in_list']

        for t_global in t_list:
            time_index_to_linear_feature_indices[t_global].append(i_linear_ind)
            linear_ind_to_gt_ind[i_linear_ind] = name2int_neuron_and_tracklet(name)
            linear_ind_to_raw_neuron_ind[i_linear_ind] = int(df_this_neuron[t_global])
            i_linear_ind += 1
        X.append(vols_array)
    return linear_ind_to_gt_ind, linear_ind_to_raw_neuron_ind, time_index_to_linear_feature_indices
