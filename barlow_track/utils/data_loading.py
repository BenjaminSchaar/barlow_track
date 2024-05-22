import logging

import numpy as np
from skimage.measure import regionprops
from wbfm.utils.external.utils_pandas import cast_int_or_nan


def get_bbox_data_for_volume(project_data, t, target_sz=np.array([8, 64, 64])):
    """List of 3d crops for all labeled (segmented) objects at time = t"""
    # Get a bbox for all neurons in 3d
    this_seg = project_data.raw_segmentation[t, ...]
    props = regionprops(this_seg)

    all_dat, all_bbox = [], []
    this_red = np.array(project_data.red_data[t, ...])
    sz = project_data.red_data.shape

    for p in props:
        bbox = p.bbox
        # Expand to get the neighborhood

        dat, _ = get_3d_crop_using_bbox(bbox, sz, target_sz, this_red)
        all_dat.append(dat)  # TODO: preallocate
        all_bbox.append(bbox)

    return all_dat, all_bbox


def get_bbox_data_for_volume_only_labeled(project_data, t, target_sz=np.array([8, 64, 64]), which_neurons=None):
    """
    Like get_bbox_data_for_volume, but only returns objects that have an ID in the final tracks
    Instead of returning a list of arrays, returns a dict indexed by the string name as found in project_data
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

    all_dat_dict = {}
    this_red = np.array(project_data.red_data[t, ...])
    sz = project_data.red_data.shape

    for p in props:
        this_label = p.label
        if this_label not in tracked_segs:
            continue
        bbox = p.bbox

        this_name = seg2name[this_label]
        dat, _ = get_3d_crop_using_bbox(bbox, sz, target_sz, this_red)

        all_dat_dict[this_name] = dat

    return all_dat_dict, seg2name, which_neurons


def get_3d_crop_using_bbox(bbox, sz, target_sz, this_red):
    """
    A real bbox does not need to be passed. Alternative is just the centroid in this 6-value format:
        zxyzxy

    Parameters
    ----------
    bbox
    sz - size of full video (4d)
    target_sz - size of output crop (3d)
    this_red - array of video. Must be slice-indexable

    Returns
    -------

    """
    z_mean = int((bbox[0] + bbox[3]) / 2)
    z0 = np.clip(z_mean - int(target_sz[0] / 2), a_min=0, a_max=sz[1])
    z1 = np.clip(z_mean + int(target_sz[0] / 2), a_min=0, a_max=sz[1])
    if z1 - z0 > target_sz[0]:
        z1 = z0 + target_sz[0]
    x_mean = int((bbox[1] + bbox[4]) / 2)
    x0 = np.clip(x_mean - int(target_sz[1] / 2), a_min=0, a_max=sz[2])
    x1 = np.clip(x_mean + int(target_sz[1] / 2), a_min=0, a_max=sz[2])
    if x1 - x0 > target_sz[1]:
        x1 = x0 + target_sz[1]
    y_mean = int((bbox[2] + bbox[5]) / 2)
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