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
def barlow_project(sample_dataset, barlow_model_path) -> BarlowProject:
    # Initialize BarlowProject with the sample dataset (old style)
    barlow_project = initialize_barlow_project_from_project_config(sample_dataset)
    assert barlow_project.target_sz is None  # Originally None, set after loading the model
    
    barlow_project.load_model(barlow_model_path)
    assert barlow_project.model is not None
    assert isinstance(barlow_project.model, BarlowTwins3d)
    # Now this is set
    assert barlow_project.target_sz is not None

    return barlow_project

# Test the BarlowProject initialization
def test_barlow_project_initialization(barlow_project: BarlowProject):
    assert isinstance(barlow_project, BarlowProject)
    # Test fields
    assert barlow_project.results_folder is not None
    assert barlow_project.logger is not None
    assert barlow_project.num_frames > 0
    assert barlow_project.segmentation_metadata is not None
    

# Test the BarlowProject methods
def test_dummy_model_run(barlow_project: BarlowProject, barlow_model_path: str):
   
    # Running the model on dummy data
    dummy_data = torch.tensor(np.random.rand(*barlow_project.target_sz), dtype=torch.float32).to(barlow_project.gpu)
    # Expects 5d tensor: (batch_size, channels, depth, height, width)
    dummy_data = dummy_data.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    predictions = barlow_project.model.embed(dummy_data).cpu().detach().numpy()

    assert predictions is not None
    # Check the size of the embedding space
    latent_size = int(barlow_project.args.projector.split('-')[-1])
    assert predictions.shape == (1, latent_size)


def test_full_model_run(barlow_project: BarlowProject):
    # Step 1: embed the test dataset
    barlow_project.embed_data()

    # Step 2: Build metadata
    barlow_project.build_embedding_metadata()

    assert barlow_project.linear_ind_to_gt_ind is not None
    assert barlow_project.linear_ind_to_raw_neuron_ind is not None
    assert barlow_project.time_index_to_linear_feature_indices is not None
    assert barlow_project.X is not None

    latent_size = int(barlow_project.args.projector.split('-')[-1])
    assert barlow_project.X.shape[1] == latent_size  # shape[0] is the number of neurons, which is variable
    num_neurons = barlow_project.X.shape[0]

    # Step 3: Track neurons via clustering
    barlow_project.track_via_clustering()
    assert barlow_project.df_tracks is not None
    assert barlow_project.df_tracks.shape[0] == barlow_project.num_frames
    assert barlow_project.df_tracks.shape[1] == num_neurons  # This is the number of neurons (variable)

    # Step 3b: Make sure the intermediate results are saved
    fnames = barlow_project._generate_filenames()
    for fname in fnames.values():
        assert fname.exists(), f"File {fname} does not exist in results folder."
