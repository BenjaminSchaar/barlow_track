"""
Shared utilities for Barlow training scripts.

Extracted from train_barlow_clusterer.py patterns to avoid duplication
between single-recording and cross-recording training.

Note: train_barlow_clusterer.py is NOT modified — this module is standalone.
"""

import logging
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

logger = logging.getLogger(__name__)


def load_config(config_fname: str) -> SimpleNamespace:
    """Load YAML config file and return as SimpleNamespace."""
    from ruamel.yaml import YAML
    with open(config_fname, "r") as f:
        cfg = YAML().load(f)
    cfg["config_fname"] = config_fname
    cfg["project_dir"] = str(Path(config_fname).parent)
    return SimpleNamespace(**cfg)


def initialize_model(args, gpu):
    """
    Create or load a BarlowTwins3d model with ResidualEncoder3D backbone.

    Parameters
    ----------
    args : SimpleNamespace
        Must contain: embedding_dim, projector, lambd, lambd_obj, target_sz_z, target_sz_xy
        Optional: pretrained_model_path, backbone_kwargs
    gpu : torch.device

    Returns
    -------
    model : BarlowTwins3d on gpu
    args : SimpleNamespace (may be updated if loading pretrained)
    """
    from barlow_track.utils.barlow import BarlowTwins3d, load_barlow_model
    from barlow_track.utils.siamese import ResidualEncoder3D

    target_sz = np.array([args.target_sz_z, args.target_sz_xy, args.target_sz_xy])
    pretrained_model_path = getattr(args, "pretrained_model_path", None)

    if pretrained_model_path is not None:
        logger.info(f"Loading pretrained model from {pretrained_model_path}")
        gpu, model, pretrained_args = load_barlow_model(pretrained_model_path)
        args.embedding_dim = pretrained_args.embedding_dim
        for k, v in vars(args).items():
            setattr(model.args, k, v)
    else:
        try:
            user_args = vars(vars(args).get("backbone_kwargs", dict()))
        except TypeError:
            user_args = dict()

        backbone_kwargs = dict(
            in_channels=1,
            num_levels=user_args.get("num_levels", 2),
            f_maps=user_args.get("f_maps", 4),
            crop_sz=target_sz,
        )
        model = BarlowTwins3d(
            args, backbone=ResidualEncoder3D, **backbone_kwargs
        ).to(gpu)

    return model, args


def setup_wandb(args):
    """Initialize wandb run if configured. Returns run or None."""
    wandb_name = getattr(args, "wandb_name", None)
    wandb_username = getattr(args, "wandb_username", None)

    if wandb_name and wandb_username:
        import wandb
        wandb.login()
        wandb_opt = dict(mode="disabled") if getattr(args, "DEBUG", False) else {}
        run = wandb.init(
            project=wandb_name, entity=wandb_username, config=args, **wandb_opt
        )
        wandb.config.update(args)
        return run
    return None


def get_gpu():
    """Return the appropriate torch device."""
    cuda_index = os.getenv("CUDA_VISIBLE_DEVICES", 0)
    return torch.device(
        f"cuda:{cuda_index}" if torch.cuda.is_available() else "cpu"
    )


def format_vectors_on_gpu(y1, y2, gpu):
    """
    Move augmented pair to GPU with correct shape.

    Handles the batch dimension transpose needed when data comes from
    the existing DataLoader (batch_size=1 wrapping).

    Expected output shape: (N_neurons, 1, z, x, y)
    """
    # If from existing DataLoader with batch_size=1: (1, N, z, x, y) -> (N, 1, z, x, y)
    if y1.dim() == 5 and y1.shape[0] == 1:
        y1 = torch.transpose(y1, 0, 1)
        y2 = torch.transpose(y2, 0, 1)
    # If from cross_recording_dataset batch builders: already (N, z, x, y)
    # Add channel dim if missing
    if y1.dim() == 4:
        y1 = y1.unsqueeze(1)
        y2 = y2.unsqueeze(1)

    y1 = y1.type(torch.FloatTensor).to(gpu)
    y2 = y2.type(torch.FloatTensor).to(gpu)
    return y1, y2
