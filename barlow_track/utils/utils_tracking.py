import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd
from backports.cached_property import cached_property
from barlow_track.utils.barlow_visualize import plot_clusters
from tqdm.auto import tqdm
from hdbscan import HDBSCAN
import hdbscan

from wbfm.utils.external.utils_pandas import fill_missing_indices_with_nan
from wbfm.utils.neuron_matching.utils_candidate_matches import rename_columns_using_matching, \
    combine_dataframes_using_mode, combine_and_rename_multiple_dataframes
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.projects.utils_neuron_names import int2name_neuron


@dataclass
class WormTsneTracker:
    X_svd: np.array
    time_index_to_linear_feature_indices: dict
    linear_ind_to_raw_neuron_ind: dict = None

    n_clusters_per_window: int = 5
    n_volumes_per_window: int = 120
    tracker_stride: int = None

    cluster_directly_on_svd_space: bool = True  # i.e. do not use tsne
    opt_tsne: dict = None
    opt_db: dict = None
    svd_components: int = 50

    # Saving info after the tracking is done
    df_global: pd.DataFrame = None
    global_clusterer: callable = None
    df_final: pd.DataFrame = None

    verbose: int = 1

    def __post_init__(self):
        # Parameters should be optimized
        self.opt_tsne = dict(n_components=2, perplexity=10, early_exaggeration=200, force_magnify_iters=500)
        self.opt_db = dict(min_cluster_size=int(0.6*self.n_volumes_per_window),
                           min_samples=int(0.1*self.n_volumes_per_window),
                           max_cluster_size=int(1.1*self.n_volumes_per_window),
                           cluster_selection_method='leaf')

        if self.tracker_stride is None:
            self.tracker_stride = int(0.5 * self.n_volumes_per_window)

        if self.n_volumes_per_window > self.num_frames:
            logging.warning(f"n_volumes_per_window ({self.n_volumes_per_window}) is greater than num_frames "
                            f"({self.num_frames}); setting to num_frames.")
            self.n_volumes_per_window = self.num_frames

        if self.verbose >= 1:
            print("Successfully initialized!")

    @property
    def global_vol_ind(self):
        return np.linspace(0, self.num_frames, self.n_volumes_per_window, dtype=int, endpoint=False)

    @property
    def num_frames(self):
        return len(self.time_index_to_linear_feature_indices)

    @property
    def all_start_volumes(self):
        all_start_volumes = list(np.arange(0, self.num_frames - self.n_volumes_per_window, step=self.tracker_stride))
        all_start_volumes.append(self.num_frames - self.n_volumes_per_window - 1)
        return all_start_volumes

    def cluster_obj2dataframe(self, db_svd, start_volume: int = None, vol_ind: list = None):
        """
        Associate cluster label ids to a (time, local ind) tuple
        i.e. build a dict
        Note: the dict key should be a tuple of (neuron_name, 'raw_neuron_ind_in_list'),
          because we want it to be a multilevel dataframe

        Parameters
        ----------
        db_svd - list or cluster object
        start_volume - optional; start of window
        vol_ind - optional; explicit indices

        Returns
        -------

        """
        if isinstance(db_svd, (list, np.ndarray)):
            labels = db_svd
        else:
            labels = db_svd.labels_

        time_index_to_linear_feature_indices = self.time_index_to_linear_feature_indices
        n_vols = self.n_volumes_per_window

        cluster_dict = {}
        # Assume the labels are in sequential order
        i_current_time = 0
        if vol_ind is None:
            current_time = start_volume

            def get_next_time(_i_current_time, _current_time):
                return _i_current_time + 1, _current_time + 1

            def get_empty_col():
                tmp = np.empty(n_vols + start_volume)
                tmp[:] = np.nan
                return tmp
        else:
            def get_next_time(_i_current_time, _tmp):
                return _i_current_time + 1, vol_ind[_i_current_time + 1]
            current_time = vol_ind[i_current_time]

            def get_empty_col():
                tmp = np.empty(np.max(vol_ind) + 1)
                tmp[:] = np.nan
                return tmp

        current_global_ind = list(time_index_to_linear_feature_indices[current_time].copy())
        current_local_ind = 0

        if self.linear_ind_to_raw_neuron_ind is not None:
            all_linear_ind = self.get_linear_indices_from_time(start_volume, time_index_to_linear_feature_indices,
                                                               vol_ind)
            for i, label in enumerate(labels):
                # Determine neuron name based on class
                if label == -1:
                    continue
                else:
                    this_neuron_name = int2name_neuron(label + 1)
                    key = (this_neuron_name, 'raw_neuron_ind_in_list')

                # Initialize dataframe dict
                if key not in cluster_dict:
                    cluster_dict[key] = get_empty_col()

                # Get the linear data index of this labeled point
                linear_index = all_linear_ind[i]

                # Convert that to a time and a local segmentation
                time_in_video = self.dict_linear_index_to_time[linear_index]
                raw_neuron_ind_in_list = self.linear_ind_to_raw_neuron_ind[linear_index]

                # Save in the dataframe dict
                cluster_dict[key][time_in_video] = raw_neuron_ind_in_list

        else:
            logging.warning("Assumes the data is in time order")
            for i, label in enumerate(labels):
                global_ind = current_global_ind.pop(0)

                if label == -1:
                    # Still want to pop above
                    pass
                else:
                    this_neuron_name = int2name_neuron(label + 1)
                    key = (this_neuron_name, 'raw_neuron_ind_in_list')

                    if key not in cluster_dict:
                        cluster_dict[key] = get_empty_col()

                    if np.isnan(cluster_dict[key][current_time]):
                        # This is a numpy array
                        cluster_dict[key][current_time] = current_local_ind
                    else:
                        # TODO: For now, just ignore the second assignment
                        pass
                        # print(f"Multiple assignments found for {this_neuron_name} at t={current_time}")

                if len(current_global_ind) == 0:
                    try:
                        i_current_time, current_time = get_next_time(i_current_time, current_time)
                    except IndexError:
                        break
                    # current_time += 1
                    current_global_ind = list(time_index_to_linear_feature_indices[current_time].copy())
                    current_local_ind = 0
                else:
                    current_local_ind += 1
        df_cluster = pd.DataFrame(cluster_dict)
        return df_cluster

    @cached_property
    def dict_linear_index_to_time(self):
        dict_linear_index_to_time = {}
        for t, ind_this_time in self.time_index_to_linear_feature_indices.items():
            for i in ind_this_time:
                dict_linear_index_to_time[i] = t
        return dict_linear_index_to_time

    def cluster_single_window(self, start_volume=0, vol_ind=None, verbose=0):
        # Unpack
        time_index_to_linear_feature_indices = self.time_index_to_linear_feature_indices

        # Options
        opt_tsne = self.opt_tsne
        opt_db = self.opt_db

        # Get this window of data
        linear_ind = self.get_linear_indices_from_time(start_volume, time_index_to_linear_feature_indices, vol_ind)
        X = self.X_svd[linear_ind, :]

        # tsne + cluster
        if verbose >= 1:
            print(f"Clustering. Using svd space directly: {self.cluster_directly_on_svd_space}")
            print(f"Input data size: {X.shape}")
        if self.cluster_directly_on_svd_space:
            Y_tsne_svd = X
            db_svd = HDBSCAN(**opt_db).fit(Y_tsne_svd)
        else:
            from tsnecuda import TSNE
            tsne = TSNE(**opt_tsne)
            Y_tsne_svd = tsne.fit_transform(X)
            db_svd = HDBSCAN(**opt_db).fit(Y_tsne_svd)

        return db_svd, Y_tsne_svd, linear_ind

    def get_linear_indices_from_time(self, start_volume, time_index_to_linear_feature_indices, vol_ind):
        if vol_ind is None:
            n_vols = self.n_volumes_per_window
            vol_ind = np.arange(start_volume, start_volume + n_vols)
        linear_ind = np.hstack([time_index_to_linear_feature_indices[i] for i in vol_ind])
        linear_ind = np.array(linear_ind, dtype=int)
        return linear_ind

    def multicluster_single_window(self, start_volume=0, vol_ind=None, to_plot=False, verbose=0):
        """
        Cluster one window self.n_clusters_per_window times, and then combine for consistency

        Parameters
        ----------
        vol_ind
        start_volume

        Returns
        -------

        """
        num_clusters = self.n_clusters_per_window

        # Get all iterations
        all_raw_dfs = []
        all_tsnes = []
        all_clusters = []
        all_ind = []
        for _ in tqdm(range(num_clusters), leave=False):
            db_svd, Y_tsne_svd, linear_ind = self.cluster_single_window(start_volume, vol_ind, verbose=verbose-1)
            df = self.cluster_obj2dataframe(db_svd, start_volume, vol_ind)
            if to_plot:
                plot_clusters(db_svd, Y_tsne_svd)
            all_raw_dfs.append(df)
            all_tsnes.append(Y_tsne_svd)
            all_clusters.append(db_svd)
            all_ind.append(linear_ind)

        if num_clusters > 1:
            # Choose a base dataframe and rename all to that one
            # For now, combine as we go so that the matching gets the benefit of any overlaps (but is slower)
            # Just choosing the one with the most neurons
            i_base = np.argmax([df.shape[1] for df in all_raw_dfs])
            df_combined = combine_and_rename_multiple_dataframes(all_raw_dfs, i_base=i_base)
        else:
            df_combined = all_raw_dfs[0]

        return df_combined, (all_raw_dfs, all_clusters, all_tsnes, all_ind)

    def track_using_overlapping_windows(self):
        """
        Clusters one window, then moves by self.tracker_stride, clusters again, and combines in sequence

        See also track_using_global_clusterer

        Returns
        -------

        """

        all_start_volumes = self.all_start_volumes

        # Track a disjoint set of points for stitching, i.e. "global" tracking
        # Increase settings for this, because it should be very stable
        self.n_clusters_per_window *= 3
        df_global = self.build_global_clusterer()
        self.n_clusters_per_window = int(self.n_clusters_per_window / 3)

        # Track each window
        if self.verbose >= 1:
            print(f"Clustering {len(all_start_volumes)} windows of length {self.n_volumes_per_window}...")
        all_dfs = []
        for start_volume in tqdm(all_start_volumes, leave=False):
            with pd.option_context('mode.chained_assignment', None):
                # Fix incorrect warning
                df_window, _ = self.multicluster_single_window(start_volume)
            all_dfs.append(df_window)

        # Make them all the right shape, then iteratively rename them to the "global" dataframe
        if self.verbose >= 1:
            print(f"Combining all dataframes to common namespace")
        all_dfs = [fill_missing_indices_with_nan(df, expected_max_t=self.num_frames)[0] for df in all_dfs]
        df_global = fill_missing_indices_with_nan(df_global, expected_max_t=self.num_frames)[0]
        all_dfs_renamed = [df_global]
        for df in tqdm(all_dfs[1:], leave=False):
            df_renamed, *_ = rename_columns_using_matching(df_global, df, try_to_fix_inf=True)
            all_dfs_renamed.append(df_renamed)

        # Finally, combine
        if self.verbose >= 1:
            print("Combining final dataframes...")
        df_combined = combine_dataframes_using_mode(all_dfs_renamed)

        self.df_final = df_combined
        # Reweight confidence?

        return df_combined, all_dfs

    def track_using_global_clusterer(self):
        """
        Track using the global clusterer, which is built from the feature space of the entire dataset

        See also track_using_overlapping_windows

        Returns
        -------
        """
        if self.global_clusterer is None:
            self.build_global_clusterer()

        # Get indices to loop through, both time and linear data matrix
        vol_ind, linear_ind = [], []
        for i in range(self.num_frames):
            if i not in self.global_vol_ind:
                vol_ind.append(i)
                linear_ind.extend(self.time_index_to_linear_feature_indices[i])
        linear_ind = np.array(linear_ind, dtype=int)

        # Cluster using pre-trained clusters
        X = self.X_svd[linear_ind, :]
        test_labels, strengths = hdbscan.approximate_predict(self.global_clusterer, X)
        df_cluster = self.cluster_obj2dataframe(test_labels, vol_ind=vol_ind)

        # Combine without renaming
        df_combined = self.df_global.combine_first(df_cluster)

        self.df_final = df_combined
        return df_combined

    def build_global_clusterer(self):
        if self.verbose >= 1:
            print(f"Initial non-local clustering...")
        # Only do one clustering, because that's all we will save
        n_clusters_per_window = self.n_clusters_per_window
        self.n_clusters_per_window = 1
        self.opt_db['prediction_data'] = True
        with pd.option_context('mode.chained_assignment', None):  # Ignore a fake warning
            df_global, (all_raw_dfs, all_clusters, all_tsnes, all_ind) = \
                self.multicluster_single_window(vol_ind=self.global_vol_ind, verbose=self.verbose)
        df_global, _ = fill_missing_indices_with_nan(df_global, expected_max_t=self.num_frames)
        self.n_clusters_per_window = n_clusters_per_window

        self.df_global = df_global
        self.global_clusterer = all_clusters[0]

        return df_global


def track_using_clusters_using_config(project_config: ModularProjectConfig, DEBUG=False):
    """
    Uses tsne + hdbscan clusters on neuron feature space as a tracker

    Parameters
    ----------
    project_config

    Returns
    -------

    """

    tracking_config = project_config.get_tracking_config()

    # Track
    tracker = WormTsneTracker.load_from_config(project_config)
    df_combined, all_raw_dfs = tracker.track_using_overlapping_windows()

    # Save
    fname = "3-tracking/postprocessing/df_cluster_tracker.h5"
    tracking_config.save_data_in_local_project(fname, df_combined, also_save_csv=True)
