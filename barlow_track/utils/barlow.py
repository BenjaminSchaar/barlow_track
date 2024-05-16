# From: http://proceedings.mlr.press/v139/zbontar21a/zbontar21a.pdf
import concurrent.futures
import gc
import numpy as np
import torch
from torch import nn, optim
import torchio as tio
import torchvision.transforms as transforms
from pytorch_lightning.core.datamodule import LightningDataModule
from tqdm.auto import tqdm
from wbfm.barlow_project.utils.data_loading import get_bbox_data_for_volume, get_bbox_data_for_volume_only_labeled
from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset
from typing import Optional
from torch.utils.data.dataloader import DataLoader
from wbfm.barlow_project.utils.siamese import Siamese
import math


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins3d(nn.Module):
    def __init__(self, args, backbone=Siamese, **backbone_kwargs):
        super().__init__()
        self.args = args

        embedding_dim = args.embedding_dim
        self.backbone = backbone(embedding_dim=embedding_dim, **backbone_kwargs)
        self.backbone.fc = nn.Identity()

        # projector
        sizes = [embedding_dim] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            # layers.append(nn.BatchNorm1d(sizes[i + 1], track_running_stats=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        # self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
        # self.bn = nn.Identity()

    def embed(self, _y):
        return self.projector(self.backbone(_y))

    def forward(self, y1, y2):
        # Shape of z: neurons x features
        if not self.args.train_both_correlations:
            c = self.calculate_correlation_matrix(y1, y2)
            loss = self.original_barlow_loss(c)
            loss_original, loss_transpose = None, None
        else:
            c_features, c_objects = self.calculate_both_correlation_matrices(y1, y2)
            # Original loss
            loss_original = self.original_barlow_loss(c_features)

            # New object loss; use same lambd and additional lambd_obj
            loss_transpose = self.args.lambd_obj*self.original_barlow_loss(c_objects)
            loss = loss_original + loss_transpose

        return loss, loss_original, loss_transpose

    def original_barlow_loss(self, c_features):
        on_diag = torch.diagonal(c_features).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c_features).pow_(2).sum()
        loss_features = on_diag + self.args.lambd * off_diag
        return loss_features

    def calculate_correlation_matrix(self, y1, y2):
        z1 = self.embed(y1)
        z2 = self.embed(y2)
        # empirical cross-correlation matrix
        z1_norm = (z1 - z1.mean(0)) / z1.std(0)
        z2_norm = (z2 - z2.mean(0)) / z2.std(0)
        this_batch_sz = z1.shape[0]

        c = torch.matmul(z1_norm.T, z2_norm) / this_batch_sz  # D x D (feature space)
        return c

    def calculate_both_correlation_matrices(self, y1, y2):
        """
        Given two images (original and augmented), calculate the correlation matrices for both the feature space and
        the object space.

        Parameters
        ----------
        y1 - torch.Tensor of shape (num_objects (pseudo-batch size), batch_size (1), z, x, y)
        y2 - same as y1

        Returns
        -------

        """
        z1 = self.embed(y1)
        z2 = self.embed(y2)
        # empirical cross-correlation matrix
        z1_norm = (z1 - z1.mean(0)) / z1.std(0)
        z2_norm = (z2 - z2.mean(0)) / z2.std(0)
        this_batch_sz = z1.shape[0]
        c_features = torch.matmul(z1_norm.T, z2_norm) / this_batch_sz  # D x D (feature space)

        # empirical cross-correlation matrix
        z1_norm = ((z1.T - z1.mean(1)) / z1.std(1)).T
        z2_norm = ((z2.T - z2.mean(1)) / z2.std(1)).T
        this_num_features = z1.shape[1]
        c_objects = torch.matmul(z1_norm, z2_norm.T) / this_num_features  # N x N (object space)

        return c_features, c_objects


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


class Transform:
    def __init__(self):
        self.final_normalization = tio.RescaleIntensity(percentiles=(5, 100))

        self.transform = tio.transforms.Compose([
            # tio.RandomFlip(axes=(1, 2), p=0.1),  # Do not flip z
            tio.RandomBlur(p=0.1),
            tio.RandomAffine(degrees=(180, 0, 0), p=1.0),  # Also allows scaling
            # tio.RandomMotion(translation=1, degrees=90, p=1.0),
            # tio.RandomElasticDeformation(max_displacement=(1, 5, 5), p=0.5),
            tio.RandomNoise(p=0.5),
            # tio.ZNormalization()
            self.final_normalization
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0, 0.485, 0.456, 0.406],
            #                      std=[1, 0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            # tio.RandomFlip(axes=(1, 2), p=0.1),  # Do not flip z
            # tio.RandomBlur(p=0.0),
            tio.RandomAffine(degrees=(180, 0, 0), p=0.1),  # Also allows scaling
            # tio.RandomElasticDeformation(max_displacement=(1, 5, 5), p=0.1),
            tio.RandomNoise(p=0.1),
            # tio.ZNormalization()
            self.final_normalization
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0, 0.485, 0.456, 0.406],
            #                      std=[1, 0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        # print(x.shape)
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2

    def normalize(self, img):
        return self.final_normalization(img)


class NeuronImageWithGTDataset(Dataset):
    def __init__(self, dict_of_neurons_of_volumes, dict_of_ids_of_volumes, which_neurons):
        # Same normalization as the Transform used to train
        # Note: that was applied to crops, not full volumes
        t = tio.RescaleIntensity(percentiles=(5, 100))

        self.dict_all_volume_crops = {i: t(torch.as_tensor(this_vol.astype(float), dtype=torch.float32)) for i, this_vol in
                                      dict_of_neurons_of_volumes.items()}
        self.dict_of_ids_of_volumes = dict_of_ids_of_volumes
        self.which_neurons = which_neurons

    def __getitem__(self, idx):
        if idx not in self.dict_of_ids_of_volumes:
            raise IndexError   # Make basic looping work with pytorch
        x = torch.unsqueeze(self.dict_all_volume_crops[idx], 0)
        gt_id = self.dict_of_ids_of_volumes[idx]
        return x, gt_id

    def __len__(self):
        return len(self.dict_all_volume_crops)

    @staticmethod
    def load_from_project(project_data, num_frames, target_sz):
        project_data.project_config.logger.info("Loading dataset from project")
        if num_frames is None:
            num_frames = project_data.num_frames

        dict_of_neurons_of_volumes, dict_of_ids_of_volumes = {}, {}

        def parallel_func(_t):
            all_dat_dict, all_seg_dict, which_neurons = get_bbox_data_for_volume_only_labeled(project_data, _t,
                                                                                              target_sz=target_sz)
            keys = list(all_dat_dict.keys())  # Need to enforce ordering?
            if len(keys) > 0:
                keys.sort()
                dict_of_ids_of_volumes[_t] = keys  # strings
                dict_of_neurons_of_volumes[_t] = np.stack([all_dat_dict[k] for k in keys], 0)
            else:
                dict_of_ids_of_volumes[_t] = []
                dict_of_neurons_of_volumes[_t] = np.zeros((0, *target_sz))

        which_neurons = project_data.get_list_of_finished_neurons()[1]

        with tqdm(total=num_frames) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(parallel_func, i): i for i in list(range(num_frames))}
                for future in concurrent.futures.as_completed(futures):
                    future.result()
                    pbar.update(1)

        return NeuronImageWithGTDataset(dict_of_neurons_of_volumes, dict_of_ids_of_volumes, which_neurons)


class NeuronAugmentedImagePairDataset(Dataset):
    def __init__(self, list_of_neurons_of_volumes):
        self.all_volume_crops = [torch.from_numpy(this_vol.astype(float)) for this_vol in list_of_neurons_of_volumes]
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


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases


def print_all_on_gpu():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass


def get_crops_from_project(crop_kwargs, frames, project_data):
    list_of_neurons_of_volumes = []
    for t in tqdm(range(frames)):
        vol_dat, _ = get_bbox_data_for_volume(project_data, t, **crop_kwargs)
        vol_dat = np.stack(vol_dat, 0)
        list_of_neurons_of_volumes.append(vol_dat)
    return list_of_neurons_of_volumes
