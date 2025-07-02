# From: http://proceedings.mlr.press/v139/zbontar21a/zbontar21a.pdf
import concurrent.futures
import gc
from pathlib import Path
import numpy as np
import torch
from torch import nn, optim
import torchio as tio
import torchvision.transforms as transforms
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from wbfm.utils.general.utils_filenames import pickle_load_binary

from barlow_track.utils.siamese import Siamese
import math
import logging

from barlow_track.utils.data_loading import get_bbox_data_for_volume_with_label


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
            loss_transpose = self.original_barlow_loss(c_objects)
            loss = (1.0-self.args.lambd_obj) * loss_original + self.args.lambd_obj * loss_transpose

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
        self.final_normalization_no_copy = tio.RescaleIntensity(percentiles=(5, 100), copy=False)

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


class NeuronImageWithGTDatasetDense(Dataset):
    """
    Preloads all image data into memory, then returns it as needed. Useful for small datasets.

    See also: NeuronImageWithGTDataset for lazy loading

    """
    def __init__(self, dict_of_neurons_of_volumes, dict_of_ids_of_volumes, which_neurons):
        # In order to synchronize the normalization used
        self._transform = Transform()
        def _normalize(x):
            # Note: applied to crops, not full volumes
            t = self._transform.final_normalization_no_copy
            return t(torch.as_tensor(x.astype(float), dtype=torch.float32))

        logging.info("Normalizing data, can take a while")
        self.dict_all_volume_crops = {i: _normalize(this_vol) for i, this_vol in
                                      tqdm(dict_of_neurons_of_volumes.items())}
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
        # TODO: refactor to be a lazy loader of the cropped data
        project_data.project_config.logger.info("Loading image data from project")
        if num_frames is None:
            num_frames = project_data.num_frames

        dict_of_neurons_of_volumes, dict_of_ids_of_volumes = {}, {}

        def parallel_func(_t):
            all_dat_dict, all_seg_dict, which_neurons = get_bbox_data_for_volume_with_label(project_data, _t,
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

        return NeuronImageWithGTDatasetDense(dict_of_neurons_of_volumes, dict_of_ids_of_volumes, which_neurons)


class NeuronImageWithGTDataset(Dataset):
    """
    Lazy loaded version of NeuronImageWithGTDatasetDense

    """
    def __init__(self, project_data, num_frames, target_sz, include_untracked=False):
        # In order to synchronize the normalization used
        self._transform = Transform()
        self.num_frames = int(num_frames)
        self.project_data = project_data
        self.target_sz = target_sz
        self.which_neurons = project_data.get_list_of_finished_neurons()[1]
        self.include_untracked = include_untracked

    def _normalize(self, x):
        # Note: applied to crops, not full volumes
        t = self._transform.final_normalization_no_copy
        return t(torch.as_tensor(x.astype(float), dtype=torch.float32))

    def __getitem__(self, idx):
        if idx > self.num_frames - 1:
            raise IndexError   # Make basic looping work with pytorch
        # Get data from the lazy loader
        x, gt_id = self.get_neurons_single_volume(idx)

        # Normalize, unsqueeze, and return
        x = torch.unsqueeze(self._normalize(x), 0)

        return x, gt_id

    def __len__(self):
        return self.num_frames

    @staticmethod
    def _fix_empty_volume(neurons_in_single_volume, target_sz):
        keys = list(neurons_in_single_volume.keys())  # Need to enforce ordering?
        if len(keys) > 0:
            keys.sort()
            ids = keys  # strings
            cropped_neuron_data = np.stack([neurons_in_single_volume[k] for k in keys], 0)
        else:
            ids = []
            cropped_neuron_data = np.zeros((0, *target_sz))
        return cropped_neuron_data, ids

    def get_neurons_single_volume(self, _t):
        neurons_in_single_volume, _, _ = get_bbox_data_for_volume_with_label(self.project_data, _t,
                                                                             target_sz=self.target_sz,
                                                                             include_untracked=self.include_untracked)
        return self._fix_empty_volume(neurons_in_single_volume, self.target_sz)
    
    def __repr__(self):
        return f"NeuronImageWithGTDataset(num_frames={self.num_frames}, target_sz={self.target_sz}, " \
               f"include_untracked={self.include_untracked})"


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


def load_barlow_model(model_fname):
    """
    Loads a model directly from the weights file, and assumes the args are saved in the same folder as args.pickle

    """
    from barlow_track.utils.siamese import ResidualEncoder3D
    state_dict = torch.load(model_fname)
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Don't use gpu when tracking (bandwidth is the bottleneck?)
    # gpu = "cpu"
    logging.info(f"Using device: {gpu}")
    # Check if there are multiple .pickle files in the same folder, and give a warning if so
    for fname in Path(model_fname).parent.glob('*.pickle'):
        if fname.name != 'args.pickle':
            logging.warning(f"Found extra pickle file in model folder: {fname}... "
                            f"Make sure args.pickle is the correct file!")
    # Possible problem: multiple models saved in the same folder with different settings
    args_fname = Path(model_fname).with_name('args.pickle')
    args = pickle_load_binary(args_fname)
    logging.info(f"Loaded args from {args_fname}: {args}")
    target_sz = np.array(args.target_sz)
    backbone_kwargs = dict(in_channels=1, num_levels=2, f_maps=4, crop_sz=target_sz)
    model = BarlowTwins3d(args, backbone=ResidualEncoder3D, **backbone_kwargs).to(gpu)
    model.load_state_dict(state_dict)
    return gpu, model, args
