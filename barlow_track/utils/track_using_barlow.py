import concurrent
import pickle
from dataclasses import dataclass, field
import logging
import os
from collections import defaultdict
from pathlib import Path
from pyexpat import model
import dask.array as da
import numpy as np
import torch
import zarr

import pandas as pd
from barlow_track.utils.barlow import load_barlow_model
from barlow_track.utils.barlow_visualize import plot_relative_accuracy
from sklearn.decomposition import TruncatedSVD
from tqdm.auto import tqdm
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.general.utils_filenames import pickle_load_binary
from wbfm.utils.external.utils_neuron_names import name2int_neuron_and_tracklet
from wbfm.utils.projects.utils_redo_steps import add_metadata_to_df_raw_ind
from barlow_track.utils.utils_tracking import WormClusterTracker, get_target_size_from_args


def track_using_barlow_from_config(project_config: ModularProjectConfig,
                                   model_fname=None,
                                   results_subfolder=None,
                                   tracking_mode='global',
                                   use_projection_space=False,
                                   to_plot_relative_accuracy=False,
                                   DEBUG=False,
                                   **project_kwargs):
    """
    Tracks a project using a pretrained Barlow Twins model

    Can reuse prior steps if they have already been run. Runs in this order:
    1. Load the pretrained neural network
    2. Embed the data (volumetric images) using the neural network
    3. Build a class to organize the embeddings and the clusterer
    4. Run the clusterer and get the final tracks
    5. Calculate accuracy (if ground truth is available) and save the results

    Note that this uses the WormTsneTracker class to do the clustering (i.e. tracking)

    Parameters
    ----------
    project_config - the project configuration object, which contains the project root and other settings
    model_fname - the exact name of the model file, or the full path to the model file
        Example: /scratch/neurobiology/zimmer/wbfm/TrainedBarlow/hyperparameter_search/trial_0/resnet50-1.pth
    results_subfolder - the subfolder to save the results in, relative to the project root
    tracking_mode - Which tracking mode. Options: 'global', 'overlapping_windows', 'streaming'
    to_plot_relative_accuracy
    use_projection_space - Whether to discard the projection head when tracking
    project_kwargs - Additional keyword arguments to pass to the ProjectData constructor

    Returns
    -------

    """
    project_data = ProjectData.load_final_project_data(project_config, **project_kwargs)
    project_config = project_data.project_config

    if model_fname is None:
        model_fname = 'checkpoint_barlow_small_projector'
        project_data.logger.warning(f"Using default network name: {model_fname}")
    if results_subfolder is None:
        results_subfolder = '3-tracking/barlow_tracker'
        project_data.logger.info(f"Output subfolder for results: {results_subfolder}")

    # Check to see if the results already exist
    results_subfolder_full = project_config.resolve_relative_path(results_subfolder)

    tracker_fname = os.path.join(results_subfolder_full, 'worm_tracker_barlow.pickle')
    if Path(tracker_fname).exists():
        project_data.logger.info("Found already saved tracker, loading...")
        tracker = pickle_load_binary(tracker_fname)

    else:
        # Next try: load metadata
        embedding_fname = os.path.join(results_subfolder_full, 'embedding.zarr')
        if Path(embedding_fname).exists():
            project_data.logger.info("Found already saved embedding files, loading...")
            X = np.array(zarr.open(embedding_fname))

            fname = os.path.join(results_subfolder_full, 'time_index_to_linear_feature_indices.pickle')
            time_index_to_linear_feature_indices = pickle_load_binary(fname)
            fname = os.path.join(results_subfolder_full, 'linear_ind_to_raw_neuron_ind.pickle')
            linear_ind_to_raw_neuron_ind = pickle_load_binary(fname)

            svd_components = 50 if project_data.num_frames > 500 else int(project_data.num_frames / 10)

            opt = dict(time_index_to_linear_feature_indices=time_index_to_linear_feature_indices,
                       svd_components=svd_components,
                       cluster_directly_on_svd_space=True,
                       n_clusters_per_window=3,
                       n_volumes_per_window=120,
                       linear_ind_to_raw_neuron_ind=linear_ind_to_raw_neuron_ind)
            tracker = WormClusterTracker(X, **opt)
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

        gpu, model, args = load_barlow_model(fname)
        target_sz = get_target_size_from_args(args)
        model.eval()

        # Embed using the model
        all_embeddings = embed_using_barlow(gpu, model, project_data, target_sz, use_projection_space)

        linear_ind_to_gt_ind, linear_ind_to_t_and_seg_id, time_index_to_linear_feature_indices, X = build_embedding_metadata(
            all_embeddings, project_data)

        svd_components = 50
        X = np.vstack(X)
        # X = np.vstack([np.vstack(list(emb.values())) for emb in all_embeddings.values()])
        project_config.logger.info(f"Truncating feature space using {svd_components} PCA components "
                                   f"(original matrix size: {X.shape})")
        # Use dask to do the SVD, because it may be very very tall
        if X.shape[0] > 10000:
            chunks = (10000, X.shape[1])
            X_dask = da.from_array(X, chunks=chunks)
            u, s, v = da.linalg.svd(X_dask)
            X_svd = np.array(u[:, :svd_components].compute())
        else:
            alg = TruncatedSVD(n_components=svd_components)
            X_svd = alg.fit_transform(X)
        project_config.logger.info(f"Finished truncation")

        # Get tracker parameters from yaml file
        tracker_cfg = project_config.get_tracking_config()
        tracker_opt = dict(opt_umap=tracker_cfg.config.get('opt_umap', dict()),
                        opt_db=tracker_cfg.config.get('opt_db', dict()))

        # Save embeddings and trackers
        opt = dict(time_index_to_linear_feature_indices=time_index_to_linear_feature_indices,
                   svd_components=svd_components,
                   cluster_directly_on_svd_space=True,
                   n_clusters_per_window=3,
                   n_volumes_per_window=120,
                   linear_ind_to_t_and_seg_id=linear_ind_to_t_and_seg_id)
        opt.update(tracker_opt)

        tracker = WormClusterTracker(X_svd, **opt)
        tracker_no_svd = WormClusterTracker(X, **opt)  # This is only for debugging later

        save_intermediate_results(X, linear_ind_to_gt_ind, linear_ind_to_t_and_seg_id, project_config, project_data,
                                  time_index_to_linear_feature_indices, tracker, tracker_no_svd,
                                  subfolder=results_subfolder_full)

    # Do the clustering
    if tracking_mode == 'global':
        project_config.logger.info("Running: track_using_global_clusterer")
        df_combined = tracker.track_using_global_clusterer()
    elif tracking_mode == 'overlapping_windows':
        project_config.logger.info("Running: track_using_overlapping_windows")
        df_combined, all_dfs = tracker.track_using_overlapping_windows()
    elif tracking_mode == 'streaming':
        project_config.logger.info("Running: track_using_streaming_clusterer")
        df_combined = tracker.track_using_streaming_clusterer()

    # Add metadata stored in the project
    project_config.logger.info("Adding metadata to the final dataframe")
    df_combined = add_metadata_to_df_raw_ind(df_combined, project_data.segmentation_metadata)

    fname = os.path.join(results_subfolder, f'df_barlow_tracks.h5')
    tracking_config = project_config.get_tracking_config()
    fname = tracking_config.save_data_in_local_project(df_combined, fname,
                                                       make_sequential_filename=False, prepend_subfolder=False)

    # Also update the project config file to point to this new h5 file
    fname_local = project_config.unresolve_absolute_path(fname)
    tracking_config = project_config.get_tracking_config()
    tracking_config.config['final_3d_tracks_df'] = fname_local
    tracking_config.update_self_on_disk()

    if to_plot_relative_accuracy:
        plot_relative_accuracy(df_combined, project_data, results_subfolder_full)


def embed_using_barlow(gpu, model, project_data, target_sz, use_projection_space):
    """
    Use a trained model to project a dataset into the latent space

    use_projection_space - if True, uses the post-projection head space; most SSL approaches discard the projector (i.e. set this to False)
    """
    from barlow_track.utils.barlow import NeuronImageWithGTDataset
    num_frames = project_data.num_frames - 1  # Why am I subtracting 1?
    dataset = NeuronImageWithGTDataset(project_data, num_frames, target_sz, include_untracked=True)
    logging.info(f"Using dataset: {dataset}")
    # names = dataset.which_neurons
    all_embeddings = defaultdict(dict)
    project_data.project_config.logger.info("Embedding using Barlow model")

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        for t, (batch, ids) in tqdm(enumerate(dataset), total=len(dataset)):
            # Move entire batch to gpu initially
            batch = batch.to(gpu)

            # Parallelize the actual embedding step using concurrent futures
            def _parallel_func(name):
                idx = ids.index(name)
                crop = torch.unsqueeze(batch[:, idx, ...], 0)
                embeddings = model.embed(crop) if use_projection_space else model.backbone(crop)
                all_embeddings[name][t] = embeddings.cpu().detach().numpy()

            # no_grad is thread-local
            # https://github.com/pytorch/pytorch/issues/20528
            # with torch.no_grad():
            futures = {executor.submit(_parallel_func, n): n for n in ids}
            for future in concurrent.futures.as_completed(futures):
                future.result()

    logging.info(f"Finished embedding {len(all_embeddings)} neurons")

    return all_embeddings


def save_intermediate_results(X, linear_ind_to_gt_ind, linear_ind_to_t_and_seg_id, project_config, project_data,
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
    fname = f'{subfolder}/linear_ind_to_t_and_seg_id.pickle'
    project_data.project_config.pickle_data_in_local_project(linear_ind_to_t_and_seg_id, fname)
    fname = f'{subfolder}/linear_ind_to_gt_ind.pickle'
    project_data.project_config.pickle_data_in_local_project(linear_ind_to_gt_ind, fname)


def build_embedding_metadata(all_embeddings, project_data):
    """
    Builds the metadata for the embeddings, including the linear index to ground truth index mapping, if any
    Complexity comes because there are two ways to build embedding keys:
    1. If there is ground truth, use the neuron name
    2. If there is no ground truth, use the metadata in the previously generated embedding key, which look like:
        untracked_time_0_1234 (i.e. take the last number as the raw_neuron_ind)

    """
    project_data.project_config.logger.info("Building embedding metadata")
    # Collect metadata
    df_gt_tracks = project_data.get_final_tracks_only_finished_neurons()[0]
    X = []
    time_index_to_linear_feature_indices = defaultdict(list)
    linear_ind_to_t_and_seg_id = {}
    linear_ind_to_gt_ind = {}
    i_linear_ind = 0
    for name, vols_all_times in all_embeddings.items():
        t_list = list(vols_all_times.keys())
        vols_array = np.vstack(list(vols_all_times.values()))

        gt_ind = -1
        has_gt = False
        if df_gt_tracks is not None:
            try:
                df_this_neuron_ind = df_gt_tracks[name, 'raw_neuron_ind_in_list']
                df_this_neuron_seg = df_gt_tracks[name, 'raw_segmentation_id']
                gt_ind = name2int_neuron_and_tracklet(name)
                has_gt = True
            except KeyError:
                pass

        for t_global in t_list:
            time_index_to_linear_feature_indices[t_global].append(i_linear_ind)
            linear_ind_to_gt_ind[i_linear_ind] = gt_ind
            if has_gt:
                linear_ind_to_t_and_seg_id[i_linear_ind] = (t_global, int(df_this_neuron_ind[t_global]), int(df_this_neuron_seg[t_global]))
            else:
                # Based on an expected name like: untracked_time_0_1234_1233, where the numbers are: t, raw_neuron_ind, raw_segmentation_id
                # i.e. using segmentation_metadata.mask_index_to_i_in_array for that object
                assert 'neuron' not in name, \
                    f"Found neuron in object named: {name}; this branch should only be for untracked objects"
                linear_ind_to_t_and_seg_id[i_linear_ind] = (t_global, int(name.split('_')[-2]), int(name.split('_')[-1]))
            i_linear_ind += 1
        X.append(vols_array)
    return linear_ind_to_gt_ind, linear_ind_to_t_and_seg_id, time_index_to_linear_feature_indices, X


# Attempts to get vibe coding to work
from barlow_track.utils.barlow import load_barlow_model
from barlow_track.utils.barlow import NeuronImageWithGTDataset


@dataclass
class BarlowProject:
    results_folder: Path
    target_sz: tuple
    gpu: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model: torch.nn.Module = None
    args: dict = None
    logger: object = None
    num_frames: int = None
    segmentation_metadata: dict = None
    project_data: ProjectData = None

    # Unpacked from the project data
    df_gt_tracks: pd.DataFrame = None  # Ground truth tracks, if available

    # Fields for intermediate products
    all_embeddings: dict = field(default_factory=lambda: defaultdict(dict))
    linear_ind_to_gt_ind: dict = field(default_factory=dict)
    linear_ind_to_raw_neuron_ind: dict = field(default_factory=dict)
    time_index_to_linear_feature_indices: dict = field(default_factory=lambda: defaultdict(list))
    tracker: object = None
    X: np.ndarray = None
    # tracker_no_svd: object = None

    # Final tracks
    df_tracks: pd.DataFrame = None

    def __post_init__(self):
        self.results_folder = Path(self.results_folder)
        # Loop through intermediate products and check if they exist; if so, load them:

    def load_model(self, model_path):
        self.gpu, self.model, self.args = load_barlow_model(model_path)
        self.target_sz = self.args.target_sz
        self.model.eval()

    def embed_data(self):
        if self.all_embeddings:
            self.logger.info("Embeddings already exist. Returning existing embeddings.")
            return self.all_embeddings

        # TODO: Refactor NeuronImageWithGTDataset to not need wbfm classes
        dataset = NeuronImageWithGTDataset(self.project_data, self.num_frames - 1, self.target_sz,
                                           include_untracked=True)
        self.all_embeddings = defaultdict(dict)
        self.logger.info("Embedding using Barlow model")

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            for t, (batch, ids) in tqdm(enumerate(dataset), total=len(dataset)):
                batch = batch.to(self.gpu)

                def _parallel_func(name):
                    idx = ids.index(name)
                    crop = torch.unsqueeze(batch[:, idx, ...], 0)
                    self.all_embeddings[name][t] = self.model.embed(crop).cpu().detach().numpy()

            # no_grad is thread-local
            # https://github.com/pytorch/pytorch/issues/20528
            # with torch.no_grad():
            futures = {executor.submit(_parallel_func, n): n for n in ids}
            for future in concurrent.futures.as_completed(futures):
                future.result()

        self.logger.info(f"Finished embedding {len(self.all_embeddings)} neurons")

    def build_embedding_metadata(self):
        if self.linear_ind_to_gt_ind:
            self.logger.info("Embedding metadata already exists. Returning existing metadata.")
            return self.linear_ind_to_gt_ind, self.linear_ind_to_raw_neuron_ind, self.time_index_to_linear_feature_indices

        # Collect metadata
        df_gt_tracks = self.df_gt_tracks
        X = []
        for name, vols_all_times in self.all_embeddings.items():
            t_list = list(vols_all_times.keys())
            vols_array = np.vstack(list(vols_all_times.values()))

            gt_ind = -1
            has_gt = False
            if df_gt_tracks is not None:
                try:
                    df_this_neuron = df_gt_tracks[name, 'raw_neuron_ind_in_list']
                    gt_ind = name2int_neuron_and_tracklet(name)
                    has_gt = True
                except KeyError:
                    pass

            for t_global in t_list:
                self.time_index_to_linear_feature_indices[t_global].append(len(X))
                self.linear_ind_to_gt_ind[len(X)] = gt_ind
                if has_gt:
                    self.linear_ind_to_raw_neuron_ind[len(X)] = int(df_this_neuron[t_global])
                else:
                    # Based on an expected name like: untracked_time_0_1234, where the last number is the raw_neuron_ind
                    # i.e. using segmentation_metadata.mask_index_to_i_in_array for that object
                    assert 'neuron' not in name, \
                        f"Found neuron in object named: {name}; this branch should only be for untracked objects"
                    self.linear_ind_to_raw_neuron_ind[len(X)] = int(name.split('_')[-1])
            X.append(vols_array)

        self.X = np.vstack(X)

        return self.linear_ind_to_gt_ind, self.linear_ind_to_raw_neuron_ind, self.time_index_to_linear_feature_indices, self.X

    def track_via_clustering(self, tracking_mode='global'):
        X = self.X
        svd_components = 50
        # Use dask to do the SVD, because it may be very very tall
        if X.shape[0] > 10000:
            chunks = (10000, X.shape[1])
            X_dask = da.from_array(X, chunks=chunks)
            u, s, v = da.linalg.svd(X_dask)
            X_svd = np.array(u[:, :svd_components].compute())
        else:
            alg = TruncatedSVD(n_components=svd_components)
            X_svd = alg.fit_transform(X)

        # Save embeddings and trackers
        opt = dict(time_index_to_linear_feature_indices=self.time_index_to_linear_feature_indices,
                   svd_components=svd_components,
                   cluster_directly_on_svd_space=True,
                   n_clusters_per_window=3,
                   n_volumes_per_window=120,
                   linear_ind_to_raw_neuron_ind=self.linear_ind_to_raw_neuron_ind)
        # TODO: Modify tracker to not be worm-specific
        tracker = WormClusterTracker(X_svd, **opt)

        self.save_intermediate_results()

        # Do the clustering
        if tracking_mode == 'global':
            df_combined = tracker.track_using_global_clusterer()
        elif tracking_mode == 'overlapping_windows':
            df_combined, all_dfs = tracker.track_using_overlapping_windows()
        elif tracking_mode == 'streaming':
            df_combined = tracker.track_using_streaming_clusterer()

        self.df_tracks = df_combined


    def save_intermediate_results(self):
        subfolder = self.results_folder

        filenames = self._generate_filenames(subfolder)

        # Use pickle_load_binary or project_config.pickle_data_in_local_project if available
        # Here, using pickle_load_binary as an example; replace with your actual pickling function if needed

        with open(filenames['tracker'], 'wb') as f:
            pickle.dump(self.tracker, f)

        # Save the embedding array (self.X) to zarr
        z = zarr.open_array(filenames['embedding'], shape=self.X.shape, chunks=(10000, self.X.shape[1]))
        z[:] = self.X

        with open(filenames['time_index_to_linear_feature_indices'], 'wb') as f:
            pickle.dump(self.time_index_to_linear_feature_indices, f)
        with open(filenames['linear_ind_to_raw_neuron_ind'], 'wb') as f:
            pickle.dump(self.linear_ind_to_raw_neuron_ind, f)
        with open(filenames['linear_ind_to_gt_ind'], 'wb') as f:
            pickle.dump(self.linear_ind_to_gt_ind, f)


    def _generate_filenames(self, subfolder=None):
        if subfolder is None:
            subfolder = self.results_folder
        return {
            'tracker': f'{subfolder}/worm_tracker_barlow.pickle',
            'embedding': f'{subfolder}/embedding.zarr',
            'time_index_to_linear_feature_indices': f'{subfolder}/time_index_to_linear_feature_indices.pickle',
            'linear_ind_to_raw_neuron_ind': f'{subfolder}/linear_ind_to_raw_neuron_ind.pickle',
            'linear_ind_to_gt_ind': f'{subfolder}/linear_ind_to_gt_ind.pickle',
        }


def initialize_barlow_project_from_project_config(project_config: ModularProjectConfig):
    # TODO: refactor to not need project_data.segmentation_metadata
    # Unpack relevant data and metadata from ModularProjectConfig
    project_data = ProjectData.load_final_project_data_from_config(project_config)
    cfg = project_data.project_config.get_tracking_config()
    results_folder = os.path.join(cfg.absolute_subfolder, 'barlow_tracking')
    os.makedirs(results_folder, exist_ok=True)
    if not os.path.exists(results_folder):
        raise FileNotFoundError(f"Model folder does not exist: {results_folder}")
    if not project_data.segmentation_metadata:
        raise ValueError("Segmentation metadata is required to initialize BarlowProject")
    # Initialize BarlowProject with the model folder and other necessary parameters
    return BarlowProject(
        results_folder=results_folder ,
        target_sz=None,  # Will be set after loading the model
        logger=project_config.logger,
        num_frames=project_data.num_frames,
        segmentation_metadata=project_data.segmentation_metadata,
        project_data=project_data,
        df_gt_tracks=project_data.get_final_tracks_only_finished_neurons()[0]
    )
    