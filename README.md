# BarlowTrack
Using a modification of Barlow Twins (self-supervised learning) to track single-cell resolution microscopy data, specifically designed for C. elegans neurons.

## Installation

### Install together with pytorch

Use the conda/mamba environment in the barlow_track.yaml file. Installation using mamba is tested on:
- Rocky Linux
- Ubuntu 22.04
- Windows 11

### Install pytorch yourself

Depending on your gpu setup, it is easier for you to install pytorch yourself and then update the rest of the environment using our simplified yaml file, `barlow_track_without_torch.yaml`.
In this case, please install the following packages using instructions from their websites:
- pytorch
- torchaudio
- torchvision
- pytorch-cuda (if using conda/mamba)
- pytorch-lightning
- torchio
- torch_geometric
- torch-scatter
- torch-sparse


## Training a network

See instructions in the [project folder](barlow_track/barlow_project_template/README.md)

## Tracking a BarlowTrack network

This is organized via the sibling repository: [wbfm](https://github.com/Zimmer-lab/wbfm).

The main instructions are found here: [Running the full pipeline](https://github.com/Zimmer-lab/wbfm/blob/main/docs/running_the_pipeline.md).


# Citation

Coming soon!