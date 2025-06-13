import pytest
from pathlib import Path
import numpy as np
import torch
import torch

from barlow_track.utils.barlow import BarlowTwins3d
from .track_using_barlow import BarlowProject, initialize_barlow_project_from_project_config
from wbfm.utils.projects.finished_project_data import ProjectData


@pytest.fixture
def barlow_model_path():
    return "/lisc/scratch/neurobiology/zimmer/wbfm/TrainedBarlow/barlow_ZIM2165_Gcamp7b_worm1-2022_11_28_from_search/trial_13/resnet50.pth"

# Load a real dataset for testing
@pytest.fixture
def sample_dataset():
    # Assuming the dataset is in a specific directory
    dataset_path = Path("/home/charles/Current_work/test_projects/barlow/worm4-2-2025-03-09/project_config.yaml")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    # Load the dataset (this is a placeholder, replace with actual loading logic)
    project_data = ProjectData.load_final_project_data(dataset_path)
    return project_data

@pytest.fixture
def barlow_project(sample_dataset) -> BarlowProject:
    # Initialize BarlowProject with the sample dataset (old style)
    return initialize_barlow_project_from_project_config(sample_dataset)

# Test the BarlowProject initialization
def test_barlow_project_initialization(barlow_project: BarlowProject):
    assert isinstance(barlow_project, BarlowProject)
    # Test fields
    assert barlow_project.results_folder is not None
    assert barlow_project.target_sz is None  # Originally None, set after loading the model
    assert barlow_project.logger is not None
    assert barlow_project.num_frames > 0
    assert barlow_project.segmentation_metadata is not None
    

# Test the BarlowProject methods
def test_load_model(barlow_project: BarlowProject, barlow_model_path: str):
    # Test loading the model
    barlow_project.load_model(barlow_model_path)
    assert barlow_project.model is not None
    assert isinstance(barlow_project.model, BarlowTwins3d)
    # Now this is set
    assert barlow_project.target_sz is not None
    
    # TODO: get image of the correct size
    dummy_data = torch.tensor(np.random.rand(*barlow_project.target_sz), dtype=torch.float32).to(barlow_project.gpu)
    # Expects 5d tensor: (batch_size, channels, depth, height, width)
    dummy_data = dummy_data.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    predictions = barlow_project.model.embed(dummy_data).cpu().detach().numpy()
    assert predictions is not None
    # Check the size of the embedding space
    latent_size = int(barlow_project.args.projector.split('-')[-1])
    assert predictions.shape == (1, latent_size)
