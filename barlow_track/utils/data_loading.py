import itertools
import logging
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from skimage.measure import regionprops
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from tqdm.auto import tqdm

from wbfm.utils.external.utils_pandas import cast_int_or_nan
from wbfm.utils.projects.utils_redo_steps import correct_tracks_dataframe_using_project
from wbfm.utils.tracklets.high_performance_pandas import get_names_from_df, get_names_of_columns_that_exist_at_t
from wbfm.utils.projects.finished_project_data import ProjectData


def get_bbox_data_for_tracklet(i_tracklet, df, project_data, t_local=None, target_sz=np.array([8, 64, 64])):
    """if t_local is None, chooses a random time"""
    # Note that .at properly selects the row by index even if some rows are dropped
    track_seg_ind = df.at[i_tracklet, 'all_ind_local']
    track_time_ind = df.at[i_tracklet, 'slice_ind']

    if t_local is None:
        t_max = len(track_seg_ind)
        t_local = np.random.randint(0, t_max)

    t_global = track_time_ind[t_local]
    i_local = track_seg_ind[t_local]

    seg_local = project_data.segmentation_metadata.i_in_array_to_mask_index(t_global, i_local)

    # Get a bbox for a neuron in 3d
    this_seg = project_data.raw_segmentation[t_global, ...]
    props = regionprops(this_seg)

    p = [p for p in props if p.label == seg_local][0]

    bbox = p.bbox

    # Expand to get the neighborhood
    sz = project_data.red_data.shape

    z0 = np.clip(bbox[0] - int(target_sz[0]/4), a_min=0, a_max=sz[1])
    z1 = np.clip(bbox[3] + int(target_sz[0]/4), a_min=0, a_max=sz[1])
    if z1 - z0 > target_sz[0]:
        z1 = z0 + target_sz[0]
    x0 = np.clip(bbox[1] - int(target_sz[1]/2), a_min=0, a_max=sz[2])
    x1 = np.clip(bbox[4] + int(target_sz[1]/2), a_min=0, a_max=sz[2])
    if x1 - x0 > target_sz[1]:
        x1 = x0 + target_sz[1]
    y0 = np.clip(bbox[2] - int(target_sz[2]/2), a_min=0, a_max=sz[3])
    y1 = np.clip(bbox[5] + int(target_sz[2]/2), a_min=0, a_max=sz[3])
    if y1 - y0 > target_sz[2]:
        y1 = y0 + target_sz[2]

    dat = project_data.red_data[t_global, z0:z1, x0:x1, y0:y1]

    # Pad, if needed, to the beginning
    diff_sz = np.clip(target_sz - np.array(dat.shape), a_min=0, a_max=np.max(target_sz))
    pad_sz = list(zip(diff_sz, np.zeros(len(diff_sz), dtype=int)))
    dat = np.pad(dat, pad_sz)

    return dat, bbox


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
        logging.warning("Found no explicitly tracked neurons, assuming all are correct")
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
        bbox = p.bbox
        this_label = p.label
        if this_label not in tracked_segs:
            continue
        
        this_name = seg2name[this_label]
        dat, _ = get_3d_crop_using_bbox(bbox, sz, target_sz, this_red)

        all_dat_dict[this_name] = dat

    return all_dat_dict, seg2name, which_neurons


def get_3d_crop_using_bbox(bbox, sz, target_sz, this_red):
    """
    A real bbox does not need to be passed. Alternative is just the centroid in this 6-value format format:
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


# MAX_TRACKLET = df.shape[0]
def get_siamese_training_triplet(df: pd.DataFrame, project_data):
    rng = np.random.default_rng()
    rand_order = rng.permutation(list(df.index))

    # Two examples from same, one different
    i_tracklet0 = rand_order[0]
    dat_anchor, _ = get_bbox_data_for_tracklet(i_tracklet0, df, project_data, t_local=None)
    dat_pos, _ = get_bbox_data_for_tracklet(i_tracklet0, df, project_data, t_local=None)

    i_tracklet1 = rand_order[1]
    dat_neg, _ = get_bbox_data_for_tracklet(i_tracklet1, df, project_data, t_local=None)

    return dat_anchor, dat_pos, i_tracklet0, dat_neg, i_tracklet1


def get_one_triplet(df, project_data):
    d0, d1, label1, d2, label2 = get_siamese_training_triplet(df, project_data)
    d0, d1, d2 = torch.from_numpy(d0.astype(np.uint8)), torch.from_numpy(d1.astype(np.uint8)), torch.from_numpy(
        d2.astype(np.uint8))
    d0 = torch.unsqueeze(d0, 0)
    d0 = torch.unsqueeze(d0, 0).float() / 255.0
    d1 = torch.unsqueeze(d1, 0)
    d1 = torch.unsqueeze(d1, 0).float() / 255.0
    d2 = torch.unsqueeze(d2, 0)
    d2 = torch.unsqueeze(d2, 0).float() / 255.0
    return d0, d1, label1, d2, label2


def build_train_loader(df, project_data, max_iters=100):
    for i in range(max_iters):
        yield get_one_triplet(df, project_data)


def build_train_loader_batch(df, project_data, max_iters=100, batch_size=4):
    for i in range(max_iters):
        all_d0, all_d1, all_d2 = [], [], []
        for i2 in range(batch_size):
            d0, d1, label1, d2, label2 = get_one_triplet(df, project_data)
            all_d0.append(d0)
            all_d1.append(d1)
            all_d2.append(d2)
        yield torch.cat(all_d0, 0), torch.cat(all_d1, 0), None, torch.cat(all_d2, 0), None


def save_training_data(df, project_data, num_triplets=1000):
    # Saves to disk for use with pytorch DataLoader class
    single_loader = build_train_loader(df, project_data, max_iters=num_triplets)

    relative_dir = 'nn_training'
    out_dir = os.path.join(project_data.project_dir, relative_dir)
    Path(out_dir).mkdir(exist_ok=True)

    # all_subfolders = []
    # all_label_triplets = []
    metadata_dict = {}
    for i, (d0, d1, label1, d2, label2) in enumerate(single_loader):
        # all_label_triplets.append([label1, label1, label2])
        subfolder = os.path.join(out_dir, f"triplet_{i}")
        Path(subfolder).mkdir(exist_ok=False)
        fname1 = os.path.join(subfolder, "anchor.pt")
        torch.save(d0, fname1)
        fname2 = os.path.join(subfolder, "positive.pt")
        torch.save(d1, fname2)
        fname3 = os.path.join(subfolder, "negative.pt")
        torch.save(d2, fname3)

        metadata_dict[subfolder] = [label1, label1, label2]
        # all_subfolders.append(subfolder)
        # all_fname_triplets.append([fname1, fname2, fname3])

    fname = os.path.join(relative_dir, 'metadata.pickle')
    project_data.project_config.pickle_data_in_local_project(metadata_dict, fname)


class NeuronTripletDataset(Dataset):

    def __init__(self, training_dir, remap_labels=False):
        self.training_dir = training_dir
        with open(os.path.join(training_dir, 'metadata.pickle'), 'rb') as f:
            self.metadata_dict = pickle.load(f)
        self.subfolders = list(self.metadata_dict.keys())

        # The raw labels could be very very large, corresponding to the tracklets they came from
        self.remap_labels = remap_labels
        self.label_mapping = {}
        self.current_label = 0

    def __len__(self):
        return len(self.metadata_dict)

    def __getitem__(self, idx):
        subfolder = self.subfolders[idx]
        fnames = ["anchor.pt", "positive.pt", "negative.pt"]
        data = [torch.load(os.path.join(subfolder, f)) for f in fnames]
        data = [torch.squeeze(d, dim=0) for d in data]  # Needed for batches
        labels = self.metadata_dict[subfolder]
        if self.remap_labels:
            new_labels = []
            for label in labels:
                if label not in self.label_mapping:
                    self.label_mapping[label] = self.current_label
                    self.current_label += 1
                new_labels.append(self.label_mapping[label])
        else:
            new_labels = labels

        # expect: labels[0] == labels[1]
        return data[0], data[1], new_labels[1], data[2], new_labels[2]

#
# Use image-space input data (i.e. ORB or VGG features)
#



class AbstractNeuronImageFeaturesFromProject(Dataset):

    def __init__(self, project_data: ProjectData, transform=None):
        self.project_data = project_data

    def __len__(self):
        return len(self.project_data.num_frames)


class AbstractNeuronImageFeatures(Dataset):

    def __init__(self, all_feature_spaces, time_to_indices_dict, transform=None):
        self.transform = transform
        self.stacked_feature_spaces = torch.from_numpy(np.vstack(all_feature_spaces))

        # Note: this works even if the classes have very different lengths, but the training may need to be adjusted
        labels = []
        class_lens = []
        class_idx_starts = []
        for i, x in enumerate(all_feature_spaces):
            class_lens.append(len(x))
            class_idx_starts.append(len(labels))
            labels.extend([i] * len(x))
        self.num_classes = len(all_feature_spaces)
        self.class_lens = np.array(class_lens)
        self.class_idx_starts = np.array(class_idx_starts)

        self.len_longest_class = np.max(self.class_lens)

        self.labels = np.array(labels)

    def __len__(self):
        return len(self.labels)


class AbstractTimeAwareNeuronImageFeatures(AbstractNeuronImageFeatures):
    """Same as AbstractNeuronImageFeatures, but when looping, idx explicitly references time"""
    def __init__(self, all_feature_spaces, time_to_indices_dict, transform=None):
        super().__init__(all_feature_spaces, time_to_indices_dict, transform)

        self.time_to_indices_dict = time_to_indices_dict

    def __len__(self):
        return len(self.time_to_indices_dict.keys())


class TimePairedFullVolumeNeuronImageFeaturesDataset(AbstractTimeAwareNeuronImageFeatures):

    def __post_init__(self):
        t_list = list(range(len(self.time_to_indices_dict.keys())))
        self.time_pairs = list(itertools.combinations(t_list, 2))

    def __len__(self):
        return len(self.time_pairs)

    def __getitem__(self, idx):
        """Here, idx refers to a pair of times

        Return 2 feature spaces, and a list of matches
        """

        t0, t1 = self.time_pairs[idx]

        idx0 = self.time_to_indices_dict[t0]
        idx1 = self.time_to_indices_dict[t1]

        features0 = self.stacked_feature_spaces[idx0, :]
        label0 = self.labels[idx0]

        features1 = self.stacked_feature_spaces[idx1, :]
        label1 = self.labels[idx1]

        matches = []
        for i, class0 in enumerate(label0):
            if class0 in label1:
                matches.append([i, label1.index(class0)])

        # Pad shorter one?
        # if features0.shape[0] < self.num_classes:
        #     num_missing = self.num_classes - features0.shape[0]
        #     padding = (0, 0, 0, num_missing)
        #     features = torch.nn.functional.pad(features0, padding)
        #
        #     label = np.hstack([label0, [-1] * num_missing])

        return features0, features1, matches


class NeuronImageFeaturesDataset(AbstractNeuronImageFeatures):
    def __getitem__(self, idx):
        features = torch.unsqueeze(self.stacked_feature_spaces[idx, :], 0)
        # features = self.transform(features)
        label = self.labels[idx]
        return features, label


class PairedNeuronImageFeaturesDataset(AbstractNeuronImageFeatures):

    def get_idx_of_same_class(self, data_idx):
        class_idx = np.argmax(self.class_idx_starts > data_idx) - 1
        this_start = self.class_idx_starts[class_idx]
        this_len = self.class_lens[class_idx]
        possible_idx = list(range(this_start, this_start + this_len))
        possible_idx.remove(data_idx)  # Sample without replacement
        i_match = np.random.randint(low=0, high=len(possible_idx))
        return possible_idx[i_match]

    def __getitem__(self, idx):
        idx2 = self.get_idx_of_same_class(idx)
        features = self.stacked_feature_spaces[[idx, idx2], :]
        # features = torch.unsqueeze(self.stacked_feature_spaces[idx, :], 0)

        label = self.labels[idx]
        return features, label


class AdjacentNeuronImageFeaturesDataset(AbstractNeuronImageFeatures):

    def get_next_idx_of_same_class(self, data_idx):
        next_idx = data_idx + 1
        class_idx = np.argmax(self.class_idx_starts > data_idx) - 1
        this_len = self.class_lens[class_idx]
        if next_idx >= this_len:
            # Go back in time instead of forward
            next_idx = data_idx - 1

        return next_idx

    def __getitem__(self, idx):
        idx2 = self.get_next_idx_of_same_class(idx)
        features = self.stacked_feature_spaces[[idx, idx2], :]
        label = self.labels[idx]
        return features, label


class TripletNeuronImageFeaturesDataset(AbstractNeuronImageFeatures):

    def get_random_idx_of_same_class_from_data_idx(self, data_idx):
        class_idx = np.argmax(self.class_idx_starts > data_idx) - 1
        return self.get_random_idx_of_class(class_idx, data_idx)

    def get_random_idx_of_class(self, class_idx, data_idx=None):
        this_start = self.class_idx_starts[class_idx]
        this_len = self.class_lens[class_idx]
        possible_idx = list(range(this_start, this_start + this_len))
        if data_idx is not None:
            possible_idx.remove(data_idx)  # Sample without replacement
        i_match = np.random.randint(low=0, high=len(possible_idx))
        return possible_idx[i_match]

    def get_triplet_idx(self, data_idx):
        idx_positive = self.get_random_idx_of_same_class_from_data_idx(data_idx)
        class_random_idx = self.get_random_class_exclusive(idx_positive)
        idx_negative = self.get_random_idx_of_class(class_random_idx)

        return idx_positive, idx_negative

    def get_random_class_exclusive(self, idx_positive):
        idx_random_class = np.random.randint(0, self.num_classes - 1)
        # Disallow the same idx as the positive class
        if idx_random_class >= idx_positive:
            idx_random_class += 1
        return idx_random_class

    def __getitem__(self, idx):
        # Anchor, positive, negative
        # print(f"Getting {idx}")
        idx_positive, idx_negative = self.get_triplet_idx(idx)
        # print(f"Found {idx_positive}, {idx_negative}")
        features = self.stacked_feature_spaces[[idx, idx_positive, idx_negative], :]

        label_positive = self.labels[idx]
        label_negative = self.labels[idx_negative]
        labels = [label_positive, label_positive, label_negative]
        return features, labels


class ShatterImageFeaturesDataset(AbstractNeuronImageFeatures):

    def get_random_idx_of_same_class_from_data_idx(self, data_idx):
        class_idx = np.argmax(self.class_idx_starts > data_idx) - 1
        return self.get_random_idx_of_class(class_idx, data_idx)

    def get_random_idx_of_class(self, class_idx, data_idx=None):
        this_start = self.class_idx_starts[class_idx]
        this_len = self.class_lens[class_idx]
        possible_idx = list(range(this_start, this_start + this_len))
        if data_idx is not None:
            possible_idx.remove(data_idx)  # Sample without replacement
        i_match = np.random.randint(low=0, high=len(possible_idx))
        return possible_idx[i_match]

    def get_triplet_idx(self, data_idx):
        idx_positive = self.get_random_idx_of_same_class_from_data_idx(data_idx)
        class_random_idx = self.get_random_class_exclusive(idx_positive)
        idx_negative = self.get_random_idx_of_class(class_random_idx)

        return idx_positive, idx_negative

    def get_random_class_exclusive(self, idx_positive):
        idx_random_class = np.random.randint(0, self.num_classes - 1)
        # Disallow the same idx as the positive class
        if idx_random_class >= idx_positive:
            idx_random_class += 1
        return idx_random_class

    def __getitem__(self, idx):
        # Anchor, positive, negative
        # print(f"Getting {idx}")
        idx_positive, idx_negative = self.get_triplet_idx(idx)
        # print(f"Found {idx_positive}, {idx_negative}")
        features = self.stacked_feature_spaces[[idx, idx_positive, idx_negative], :]

        label_positive = self.labels[idx]
        label_negative = self.labels[idx_negative]
        labels = [label_positive, label_positive, label_negative]
        return features, labels


class FullVolumeNeuronImageFeaturesDataset(AbstractNeuronImageFeatures):

    def __len__(self):
        return self.len_longest_class

    def get_indices_within_class(self, idx):
        if np.any(idx > self.class_lens):
            subset_starts = [start + idx for start, this_len in zip(self.class_idx_starts, self.class_lens) if idx < this_len]
            return np.array(subset_starts)
        else:
            return self.class_idx_starts + idx

    def __getitem__(self, idx):

        all_idx = self.get_indices_within_class(idx)

        features = self.stacked_feature_spaces[all_idx, :]
        label = self.labels[all_idx]

        if features.shape[0] < self.num_classes:
            num_missing = self.num_classes - features.shape[0]
            padding = (0, 0, 0, num_missing)
            features = torch.nn.functional.pad(features, padding)

            label = np.hstack([label, [-1] * num_missing])
        # features = torch.unsqueeze(self.stacked_feature_spaces[idx, :], 0)

        return features, label


## Utility functions
def build_ground_truth_neuron_feature_spaces(project_data: ProjectData,
                                             assume_all_neurons_correct=False,
                                             num_neurons=None, num_frames=None,
                                             col='auto-added tracklets correct'):

    all_frames = project_data.raw_frames
    df = project_data.final_tracks

    # Load ground truth
    if assume_all_neurons_correct:
        neurons_that_are_finished = get_names_from_df(df)
    else:
        tracking_cfg = project_data.project_config.get_tracking_config()
        fname = "manual_annotation/manual_tracking.csv"
        fname = tracking_cfg.resolve_relative_path(fname, prepend_subfolder=True)
        df_manual_tracking = pd.read_csv(fname)

        # Use the ones that are partially tracked as well
        neurons_that_are_finished = list(df_manual_tracking[df_manual_tracking[col]]['Neuron ID'])
        # neurons_that_are_finished = list(df_manual_tracking[df_manual_tracking['Finished?']]['Neuron ID'])

    if num_neurons is None:
        neurons = neurons_that_are_finished
    else:
        neurons = neurons_that_are_finished[:num_neurons]

    if num_frames is None:
        num_frames = project_data.num_frames - 1

    all_feature_spaces, time_to_indices_dict = feature_spaces_from_dataframe(all_frames, df, neurons, num_frames)

    return all_feature_spaces, time_to_indices_dict


def feature_spaces_from_dataframe(all_frames, df, neurons, num_frames):
    # Get stored feature spaces from Frame objects
    all_feature_spaces = []
    time_to_indices_dict = defaultdict(list)
    for neuron in tqdm(neurons):
        this_neuron = df[neuron]
        this_feature_space = []
        this_feature_times_list = []

        for t in range(num_frames):
            ind_within_frame = this_neuron['raw_neuron_ind_in_list'][t]
            if np.isnan(ind_within_frame):
                continue
            else:
                ind_within_frame = int(ind_within_frame)
            frame = all_frames[t]

            if ind_within_frame >= frame.all_features.shape[0]:
                logging.warning("Neuron not found within frame; these objects may need to be regenerated")
                continue
            this_feature_space.append(frame.all_features[ind_within_frame, :])
            this_feature_times_list.append(t)
        this_feature_space = np.vstack(this_feature_space)

        offset = len(all_feature_spaces)
        for i, t in enumerate(this_feature_times_list):
            time_to_indices_dict[t].append(i + offset)
        all_feature_spaces.append(this_feature_space)
    return all_feature_spaces, time_to_indices_dict


def build_per_tracklet_feature_spaces(project_data: ProjectData, t_template=0, num_frames=None):

    # Load ground truth
    all_frames = project_data.raw_frames
    df_tracklets = project_data.df_all_tracklets
    these_names = get_names_of_columns_that_exist_at_t(df_tracklets, t_template)
    if num_frames is None:
        num_frames = project_data.num_frames - 1

    all_feature_spaces, time_to_indices_dict = feature_spaces_from_dataframe(all_frames, df_tracklets, these_names, num_frames)

    return all_feature_spaces, time_to_indices_dict


def get_test_train_split(project_data: ProjectData, num_neurons=None, num_frames=None,
                         batch_size=32, train_fraction=0.8):

    all_feature_spaces, time_to_indices_dict = build_ground_truth_neuron_feature_spaces(project_data,
                                                                                         num_neurons, num_frames)
    alldata = NeuronImageFeaturesDataset(all_feature_spaces)

    train_fraction = int(len(alldata) * train_fraction)
    splits = [train_fraction, len(alldata) - train_fraction]
    trainset, testset = random_split(alldata, splits)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    return trainloader, testloader


class NeuronImageFeaturesDataModule(LightningDataModule):
    """Return neurons and their labels, e.g. for a classifier"""
    def __init__(self, batch_size=64, project_data: ProjectData = None, num_neurons=None, num_frames=None,
                 train_fraction=0.8, val_fraction=0.1, base_dataset_class=NeuronImageFeaturesDataset,
                 assume_all_neurons_correct=False):
        super().__init__()
        self.batch_size = batch_size
        self.project_data = project_data
        self.num_neurons = num_neurons
        self.num_frames = num_frames
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.base_dataset_class = base_dataset_class
        self.assume_all_neurons_correct = assume_all_neurons_correct

    def setup(self, stage: Optional[str] = None):
        # transform and split
        all_feature_spaces, time_to_indices_dict = build_ground_truth_neuron_feature_spaces(self.project_data,
                                                                      num_neurons=self.num_neurons,
                                                                      num_frames=self.num_frames,
                                                                      assume_all_neurons_correct=self.assume_all_neurons_correct)
        alldata = self.base_dataset_class(all_feature_spaces, time_to_indices_dict)

        train_fraction = int(len(alldata) * self.train_fraction)
        val_fraction = int(len(alldata) * self.val_fraction)
        splits = [train_fraction, val_fraction, len(alldata) - train_fraction - val_fraction]
        trainset, valset, testset = random_split(alldata, splits)

        # assign to use in dataloaders
        self.train_dataset = trainset
        self.val_dataset = valset
        self.test_dataset = testset

        self.alldata = alldata

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class NeuronImageFeaturesDataModuleFromProject(LightningDataModule):
    """Return neurons and their labels, e.g. for a classifier"""
    def __init__(self, batch_size=64, project_data: ProjectData = None, num_neurons=None, num_frames=None,
                 train_fraction=0.8, val_fraction=0.1, base_dataset_class=AbstractNeuronImageFeaturesFromProject,
                 assume_all_neurons_correct=False, dataset_kwargs=None):
        super().__init__()
        if dataset_kwargs is None:
            dataset_kwargs = {}
        self.batch_size = batch_size
        self.project_data = project_data
        self.num_neurons = num_neurons
        self.num_frames = num_frames
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.base_dataset_class = base_dataset_class
        self.dataset_kwargs = dataset_kwargs
        self.assume_all_neurons_correct = assume_all_neurons_correct

    def setup(self, stage: Optional[str] = None):
        # transform and split
        alldata = self.base_dataset_class(self.project_data, **self.dataset_kwargs)

        train_fraction = int(len(alldata) * self.train_fraction)
        val_fraction = int(len(alldata) * self.val_fraction)
        splits = [train_fraction, val_fraction, len(alldata) - train_fraction - val_fraction]
        trainset, valset, testset = random_split(alldata, splits)

        # assign to use in dataloaders
        self.train_dataset = trainset
        self.val_dataset = valset
        self.test_dataset = testset

        self.alldata = alldata

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class TrackletImageFeaturesDataModule(LightningDataModule):
    """Same as NeuronImageFeaturesDataModule, but for tracklets not tracks"""
    def __init__(self, batch_size=256, project_data: ProjectData=None, t_template=0, num_frames=None,
                 train_fraction=0.8, val_fraction=0.1, base_dataset_class=TripletNeuronImageFeaturesDataset):
        super().__init__()
        self.batch_size = batch_size
        self.project_data = project_data
        self.t_template = t_template
        self.num_frames = num_frames
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.base_dataset_class = base_dataset_class

    def setup(self, stage: Optional[str] = None):
        # transform and split
        all_feature_spaces, time_to_indices_dict = build_per_tracklet_feature_spaces(self.project_data, num_frames=self.num_frames,
                                                               t_template=self.t_template)
        alldata = self.base_dataset_class(all_feature_spaces, time_to_indices_dict)

        train_fraction = int(len(alldata) * self.train_fraction)
        val_fraction = int(len(alldata) * self.val_fraction)
        splits = [train_fraction, val_fraction, len(alldata) - train_fraction - val_fraction]
        trainset, valset, testset = random_split(alldata, splits)

        # assign to use in dataloaders
        self.train_dataset = trainset
        self.val_dataset = valset
        self.test_dataset = testset

        self.alldata = alldata

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class SequentialLoader:
    """
    Dataloader wrapper around multiple dataloaders, that returns data from them in sequence

    From: https://github.com/PyTorchLightning/pytorch-lightning/issues/12650
    """

    def __init__(self, *dataloaders: DataLoader):
        self.dataloaders = dataloaders

    def __len__(self):
        return sum(len(d) for d in self.dataloaders)

    def __iter__(self):
        for dataloader in self.dataloaders:
            yield from dataloader


class NeuronImageFeaturesDataModuleFromMultipleProjects(LightningDataModule):
    """Return neurons and their labels, e.g. for a classifier"""
    def __init__(self, batch_size=64, all_project_data: List[ProjectData] = None, num_neurons=None, num_frames=None,
                 train_fraction=0.8, val_fraction=0.1, base_dataset_class=AbstractNeuronImageFeaturesFromProject,
                 assume_all_neurons_correct=False, dataset_kwargs=None):
        super().__init__()
        if dataset_kwargs is None:
            dataset_kwargs = {}
        self.batch_size = batch_size
        self.all_project_data = all_project_data
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.base_dataset_class = base_dataset_class
        self.dataset_kwargs = dataset_kwargs
        self.assume_all_neurons_correct = assume_all_neurons_correct

    def setup(self, stage: Optional[str] = None):
        # Split each individually
        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []
        self.alldata = []
        for project_data in self.all_project_data:
            alldata = self.base_dataset_class(project_data, **self.dataset_kwargs)

            train_fraction = int(len(alldata) * self.train_fraction)
            val_fraction = int(len(alldata) * self.val_fraction)
            splits = [train_fraction, val_fraction, len(alldata) - train_fraction - val_fraction]
            trainset, valset, testset = random_split(alldata, splits)

            # assign to use in dataloaders
            self.train_dataset.append(trainset)
            self.val_dataset.append(valset)
            self.test_dataset.append(testset)

            self.alldata.append(alldata)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dataloaders = (
            DataLoader(
                dataset=dataset,
                batch_size=self.batch_size
            )
            for dataset in self.train_dataset
        )
        return SequentialLoader(*dataloaders)

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        dataloaders = (
            DataLoader(
                dataset=dataset,
                batch_size=self.batch_size
            )
            for dataset in self.val_dataset
        )
        return SequentialLoader(*dataloaders)

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        dataloaders = (
            DataLoader(
                dataset=dataset,
                batch_size=self.batch_size
            )
            for dataset in self.test_dataset
        )
        return SequentialLoader(*dataloaders)


def load_data_with_ground_truth():
    ## Load the 4 datasets that have manual annotations
    folder_name = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/"
    fname = os.path.join(folder_name, "round1_worm1/project_config.yaml")
    project_data1 = ProjectData.load_final_project_data_from_config(fname, to_load_frames=True)
    fname = os.path.join(folder_name, "round1_worm4/project_config.yaml")
    project_data2 = ProjectData.load_final_project_data_from_config(fname, to_load_frames=True)
    fname = os.path.join(folder_name, "round2_worm6/project_config.yaml")
    project_data3 = ProjectData.load_final_project_data_from_config(fname, to_load_frames=True)
    fname = os.path.join(folder_name, "round2_worm3/project_config.yaml")
    project_data4 = ProjectData.load_final_project_data_from_config(fname, to_load_frames=True)
    ## Confirm that the tracks are correct
    df1 = correct_tracks_dataframe_using_project(project_data1, overwrite=False, actually_save=False)
    project_data1.final_tracks = df1
    df2 = correct_tracks_dataframe_using_project(project_data2, overwrite=False, actually_save=False)
    project_data2.final_tracks = df2
    df3 = correct_tracks_dataframe_using_project(project_data3, overwrite=False, actually_save=False)
    project_data3.final_tracks = df3
    df4 = correct_tracks_dataframe_using_project(project_data4, overwrite=False, actually_save=False)
    project_data4.final_tracks = df4
    ## Align with the manual annotation .csv file
    project_data1.finished_neurons_column_name = 'Finished?'  # round1 worm 1
    project_data2.finished_neurons_column_name = 'Finished?'  # round1 worm 4
    project_data3.finished_neurons_column_name = 'first 100 frames'  # round2 worm 6
    project_data4.finished_neurons_column_name = 'Finished?'  # round2 worm 3
    project_data1._custom_frame_indices = list(
        range(1000, 3000))  # round1 worm 1; do not include the non-moving portion
    project_data3.num_frames = 100  # round2 worm 6

    all_project_data = [project_data1, project_data2, project_data3, project_data4]

    return all_project_data
