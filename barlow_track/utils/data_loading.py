import logging
import pandas as pd
import numpy as np
from skimage.measure import regionprops
from wbfm.utils.external.utils_pandas import cast_int_or_nan


def get_bbox_data_for_volume(project_data, t, target_sz=np.array([8, 64, 64]), raise_if_no_neurons=False):
    """List of 3d crops for all labeled (segmented) objects at time = t"""
    # Get a bbox for all neurons in 3d
    this_seg = project_data.raw_segmentation
    if this_seg is None:
        # Then we try to use the centroids directly
        df_tracks = project_data.intermediate_global_tracks
        _get_bbox = lambda i, neuron: df_tracks.loc[t, (neuron, ['z', 'x', 'y'])].values
        neurons = df_tracks.columns.get_level_values(0).unique()
    else:
        props = regionprops(this_seg[t, ...])
        neurons = np.arange(len(props))
        _get_bbox = lambda i, neuron: props[i].bbox

    all_dat, all_bbox = [], []
    this_red = np.array(project_data.red_data[t, ...])
    sz = project_data.red_data.shape

    for i, neuron in enumerate(neurons):
        try:
            bbox_or_centroid = _get_bbox(i, neuron)
        except (IndexError, KeyError) as e:
            logging.warning(f"Could not get bbox for neuron {neuron} at time {t}, skipping")
            if raise_if_no_neurons:
                raise e
            continue
        if np.isnan(bbox_or_centroid[0]):
            continue

        # Expand to get the neighborhood
        dat, _ = get_3d_crop_using_bbox_or_centroid(bbox_or_centroid, sz, target_sz, this_red)
        all_dat.append(dat)  # TODO: preallocate
        all_bbox.append(bbox_or_centroid)

    return all_dat, all_bbox


def get_bbox_data_for_volume_with_label(project_data, t, target_sz=np.array([8, 64, 64]), which_neurons=None,
                                        include_untracked=False):
    """
    Like get_bbox_data_for_volume, but only returns objects that have an ID in the final tracks (unless include_untracked=True)
    Instead of returning a list of arrays, returns a dict indexed by the string name as found in project_data
    """
    if which_neurons is None:
        which_neurons = project_data.finished_neuron_names()
    if which_neurons is None:
        project_data.project_config.logger.warning("Found no explicitly tracked neurons, assuming all are correct")
        which_neurons = project_data.neuron_names

    # Get the tracked mask indices, with a mapping from their neuron name
    name2seg = {}
    if project_data.final_tracks is not None:
        try:
            name2seg = dict(project_data.final_tracks.loc[t, (slice(None), 'raw_segmentation_id')].droplevel(1))
        except KeyError:
            logging.warning(f"Could not find any tracked neurons at time {t}, ignoring labels")

    tracked_segs = project_data.finished_neuron_names()
    seg2name = {}
    for k, v in name2seg.items():
        seg2name[cast_int_or_nan(v)] = k
    # tracked_segs = set(seg2name.keys())

    # Get a bbox for all neurons in 3d, but optionally skip the untracked mask indices
    all_dat_dict = {}
    this_red = project_data.red_data[t, ...]
    # Check for dask arrays
    if hasattr(this_red, 'compute'):
        this_red = this_red.compute()
    sz = project_data.red_data.shape

    # Use the metadata as calculated in the project
    try:
        row_data, column_names = project_data.segmentation_metadata.get_all_neuron_metadata_for_single_time(t, as_dataframe=False)
        mdata = pd.DataFrame(dict(zip(column_names, row_data)))
    except FileNotFoundError as e:
        # Fallback to the intermediate tracks
        if project_data.intermediate_global_tracks is None:
            raise e
        else:
            mdata = project_data.intermediate_global_tracks.loc[t].unstack(level=1).dropna()
        if include_untracked:
            logging.warning("Could not find raw segmentation metadata, so cannot include untracked neurons "
                            "- returning only objects in intermediate_global_tracks")

    for i, row in mdata.iterrows():
        this_seg_label = int(row['raw_segmentation_id'])
        if this_seg_label in tracked_segs:
            this_name = seg2name[this_seg_label]
        else:
            if not include_untracked:
                continue
            else:
                # Make a unique name for this untracked object, but keep the correct label
                ind_in_list = project_data.segmentation_metadata.mask_index_to_i_in_array(t, this_seg_label)
                this_name = f"untracked_time_{t}_{ind_in_list:04d}_{this_seg_label:04d}"
        zxy = [row['z'], row['x'], row['y']]
        # Repeat to be zxyzxy
        zxyzxy = [zxy[0], zxy[1], zxy[2], zxy[0], zxy[1], zxy[2]]
        dat, _ = get_3d_crop_using_bbox_or_centroid(zxyzxy, sz, target_sz, this_red)
        all_dat_dict[this_name] = dat

    return all_dat_dict, seg2name, which_neurons


def get_bbox_data_for_volume_lazy(project_data, t, target_sz=np.array([8, 64, 64]), which_neurons=None):
    """
    Like get_bbox_data_for_volume, but returns a generator instead of a dictionary of arrays in memory
    """

    if which_neurons is None:
        which_neurons = project_data.finished_neuron_names()
    if which_neurons is None:
        project_data.project_config.logger.warning("Found no explicitly tracked neurons, assuming all are correct")
        which_neurons = project_data.neuron_names

    # Get the tracked mask indices, with a mapping from their neuron name
    name2seg = dict(project_data.final_tracks.loc[t, (slice(None), 'raw_segmentation_id')].droplevel(1))
    seg2name = {}
    for k, v in name2seg.items():
        seg2name[cast_int_or_nan(v)] = k
    tracked_segs = set(seg2name.keys())

    # Get a bbox for all neurons in 3d, but skip the untracked mask indices
    this_seg = project_data.raw_segmentation[t, ...]
    props = regionprops(this_seg)

    this_red = np.array(project_data.red_data[t, ...])
    sz = project_data.red_data.shape

    for p in props:
        this_label = p.label
        if this_label not in tracked_segs:
            continue
        bbox = p.bbox

        this_name = seg2name[this_label]
        dat, _ = get_3d_crop_using_bbox_or_centroid(bbox, sz, target_sz, this_red)

        yield this_name, dat


def get_3d_crop_using_bbox_or_centroid(zxyzxy, sz, target_sz, this_red):
    """
    A real bbox does not need to be passed. Alternative is just the centroid in zxy format

    Parameters
    ----------
    zxyzxy
    sz - size of full video (4d)
    target_sz - size of output crop (3d)
    this_red - array of video. Must be slice-indexable

    Returns
    -------

    """
    if len(zxyzxy) == 3:
        z_mean = int(np.round(zxyzxy[0]))
        x_mean = int(np.round(zxyzxy[1]))
        y_mean = int(np.round(zxyzxy[2]))
    elif len(zxyzxy) == 6:
        zxyzxy 
        z_mean = int((zxyzxy[0] + zxyzxy[3]) / 2)
        x_mean = int((zxyzxy[1] + zxyzxy[4]) / 2)
        y_mean = int((zxyzxy[2] + zxyzxy[5]) / 2)
    else:
        raise ValueError(f"Unknown bbox or centroid format; {zxyzxy}")

    z0 = np.clip(z_mean - int(target_sz[0] / 2), a_min=0, a_max=sz[1])
    z1 = np.clip(z_mean + int(target_sz[0] / 2), a_min=0, a_max=sz[1])
    if z1 - z0 > target_sz[0]:
        z1 = z0 + target_sz[0]
    x0 = np.clip(x_mean - int(target_sz[1] / 2), a_min=0, a_max=sz[2])
    x1 = np.clip(x_mean + int(target_sz[1] / 2), a_min=0, a_max=sz[2])
    if x1 - x0 > target_sz[1]:
        x1 = x0 + target_sz[1]
    y0 = np.clip(y_mean - int(target_sz[2] / 2), a_min=0, a_max=sz[3])
    y1 = np.clip(y_mean + int(target_sz[2] / 2), a_min=0, a_max=sz[3])
    if y1 - y0 > target_sz[2]:
        y1 = y0 + target_sz[2]
    dat = this_red[z0:z1, x0:x1, y0:y1]
    # Pad, if needed, to the beginning
    diff_sz = np.clip(target_sz - np.array(dat.shape), a_min=0, a_max=np.max(target_sz))
    pad_sz = list(zip(diff_sz, np.zeros(len(diff_sz), dtype=int)))
    dat = np.pad(dat, pad_sz)
    new_bbox = [z0, x0, y0, z1, x1, y1]
    return dat, new_bbox