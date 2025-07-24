from typing import Optional

import numpy as np
import torch
import random
from barlow_track.utils.barlow import Transform
from barlow_track.utils.data_loading import get_bbox_data_for_volume
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, random_split, DataLoader
from tqdm.auto import tqdm


class NeuronAugmentedImagePairDataset(Dataset):
    def __init__(self, list_of_neurons_of_volumes):
        self.all_volume_crops = []
        for neuron in list_of_neurons_of_volumes:
            self.all_volume_crops.append(torch.from_numpy(neuron.astype(np.float32)))
        self.augmentor = Transform()

    def __getitem__(self, idx):
        _idx = self.idx_biggest_to_smallest()[idx]

        crops = torch.unsqueeze(self.all_volume_crops[_idx], 0)
        # Assume batch=1
        y1, y2 = self.augmentor(torch.squeeze(crops))

        # Normalize; different batch each time
        # sz = y1.shape[0]  # Todo: set this to a global mean and std
        # n = nn.InstanceNorm3d(sz, affine=False)
        # y1 = n(y1)
        # y2 = n(y2)

        return y1, y2

    def idx_biggest_to_smallest(self):
        # With variable batch sizes, must to largest first for memory reasons:
        # https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741/11
        all_shapes = np.array([crop.shape[0] for crop in self.all_volume_crops])
        idx_sorted = np.argsort(-all_shapes)
        return idx_sorted

    def __len__(self):
        return len(self.all_volume_crops)


class NeuronCropImageDataModule(LightningDataModule):
    """Return neurons and their labels, e.g. for a classifier"""

    def __init__(self, batch_size=8, project_data=None, num_frames=100,
                 train_fraction=0.8, val_fraction=0.1, base_dataset_class=NeuronAugmentedImagePairDataset,
                 crop_kwargs=None):
        super().__init__()
        if crop_kwargs is None:
            crop_kwargs = {}
        self.batch_size = batch_size
        self.project_data = project_data
        self.num_frames = num_frames
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.base_dataset_class = base_dataset_class
        self.crop_kwargs = crop_kwargs

    def setup(self, stage: Optional[str] = None):
        # Get data, then build torch classes
        frames = self.num_frames
        project_data = self.project_data
        crop_kwargs = self.crop_kwargs

        list_of_neurons_of_volumes = get_crops_from_project(crop_kwargs, frames, project_data)
        alldata = self.base_dataset_class(list_of_neurons_of_volumes)

        self.list_of_neurons_of_volumes = list_of_neurons_of_volumes

        # transform and split
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


def get_crops_from_project(crop_kwargs, frames, project_data):
    list_of_neurons_of_volumes = []
    max_num_frames = project_data.num_frames
    random_sample = random.sample(range(max_num_frames), max_num_frames)
    
    i = 0
    num_selected_frames = 0
    with tqdm(total=frames, desc="Sampling frames") as pbar:
        while i < len(random_sample) and num_selected_frames < frames:
            t = random_sample[i]
            vol_dat, _ = get_bbox_data_for_volume(project_data, t, **crop_kwargs)
            if len(vol_dat) != 0 and len(vol_dat) <= 200:
                vol_dat = np.stack(vol_dat, 0)
                list_of_neurons_of_volumes.append(vol_dat)
                num_selected_frames += 1
                pbar.update(1)  # manually update progress when sample is valid
            i += 1
    
    print("Number of frames selected: " + str(len(list_of_neurons_of_volumes)))
    return list_of_neurons_of_volumes
