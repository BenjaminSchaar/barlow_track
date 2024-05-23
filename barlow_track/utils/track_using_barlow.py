import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import zarr
from barlow_track.utils.barlow_visualize import plot_relative_accuracy
from sklearn.decomposition import TruncatedSVD
from tqdm.auto import tqdm
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.general.utils_filenames import pickle_load_binary
from wbfm.utils.projects.utils_neuron_names import name2int_neuron_and_tracklet
from barlow_track.utils.track_using_clusters import WormTsneTracker


def track_using_barlow_from_config(project_config: ModularProjectConfig,
                                   model_fname=None,
                                   results_subfolder=None,
                                   use_windowed_clustering=True,
                                   to_plot_relative_accuracy=True):
    """
    Tracks a project using a pretrained Barlow Twins model

    Can reuse prior steps if they have already been run. Runs in this order:
    1. Load the pretrained neural network
    2. Embed the data (volumetric images) using the neural network
    3. Build a class to organize the embeddings and the clusterer
    4. Run the clusterer and get the final tracks
    5. Calculate accuracy and save the results

    Note that this uses the WormTsneTracker class to do the full windowed clustering and tracking

    Parameters
    ----------
    project_config
    model_fname - the exact name of the model file, or the full path to the model file
        Example: /scratch/neurobiology/zimmer/wbfm/TrainedBarlow/hyperparameter_search/trial_0/resnet50-1.pth
    results_subfolder
    use_windowed_clustering - Whether to use the windowed clustering or the global clustering
    to_plot_relative_accuracy

    Returns
    -------

    """
    if model_fname is None:
        model_fname = 'checkpoint_barlow_small_projector'
        project_config.logger.warning(f"Using default network name: {model_fname}")
    if results_subfolder is None:
        # The default folder is built from the model fname, but removes the "checkpoint_" prefix
        # Also if it is a full path, just take the last foldername
        if Path(model_fname).is_absolute():
            results_subfolder = str(Path(model_fname).parent.stem)
        else:
            results_subfolder = str(model_fname)
        results_subfolder = results_subfolder.replace('checkpoint_', '')
        results_subfolder = f'3-tracking/{results_subfolder}'
        project_config.logger.info(f"Output subfolder for results: {results_subfolder}")

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

        with torch.inference_mode():
            gpu, model, target_sz = load_barlow_model(fname)
            model.eval()

            # Embed using the model
            all_embeddings = embed_using_barlow(gpu, model, project_data, target_sz)

            linear_ind_to_gt_ind, linear_ind_to_raw_neuron_ind, time_index_to_linear_feature_indices = build_embedding_metadata(
                all_embeddings, project_data)

        svd_components = 50
        project_config.logger.info(f"Truncating feature space using {svd_components} PCA components")
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
    if use_windowed_clustering:
        project_config.logger.info("Running: track_using_windowed_clusterer")
        df_combined = tracker.track_using_windowed_clusterer()
    else:
        project_config.logger.info("Running: track_using_global_clusterer")
        df_combined = tracker.track_using_global_clusterer()

    fname = os.path.join(results_subfolder, f'df_barlow_tracks.h5')
    project_config.save_data_in_local_project(df_combined, fname, make_sequential_filename=True)

    if to_plot_relative_accuracy:
        plot_relative_accuracy(df_combined, results_subfolder, project_data)


def embed_using_barlow(gpu, model, project_data, target_sz):
    from barlow_track.utils.barlow import NeuronImageWithGTDataset
    num_frames = project_data.num_frames - 1
    dataset = NeuronImageWithGTDataset.load_from_project(project_data, num_frames, target_sz)
    names = dataset.which_neurons
    all_embeddings = defaultdict(dict)
    project_data.project_config.logger.info("Embedding using Barlow model")
    with torch.no_grad():
        for t, (batch, ids) in tqdm(enumerate(dataset), total=len(dataset)):
            for name in names:
                if name in ids:
                    idx = ids.index(name)

                    crop = torch.unsqueeze(batch[:, idx, ...], 0).to(gpu)
                    all_embeddings[name][t] = model.embed(crop).cpu().numpy()
    return all_embeddings


def load_barlow_model(model_fname):
    from barlow_track.utils.barlow import BarlowTwins3d
    from barlow_track.utils.siamese import ResidualEncoder3D
    state_dict = torch.load(model_fname)
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {gpu}")
    args = pickle_load_binary(Path(model_fname).with_name('args.pickle'))
    logging.info(f"Found args: {args}")
    target_sz = np.array([4, 128, 128])  # Should be loaded from the args
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
    project_data.project_config.logger.info("Building embedding metadata")
    # Collect metadata
    df_tracks = project_data.get_final_tracks_only_finished_neurons()[0]
    X = []
    time_index_to_linear_feature_indices = defaultdict(list)
    linear_ind_to_raw_neuron_ind = {}
    linear_ind_to_gt_ind = {}
    i_linear_ind = 0
    for name, vols_all_times in all_embeddings.items():
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
