import argparse
import pickle
import numpy as np
import torch
from wbfm.barlow_project.utils.barlow import BarlowTwins3d, NeuronCropImageDataModule
from wbfm.barlow_project.utils.barlow_visualize import visualize_model_performance
from wbfm.barlow_project.utils.siamese import ResidualEncoder3D
from wbfm.utils.projects.finished_project_data import ProjectData

if __name__ == "__main__":
    # Get args
    parser = argparse.ArgumentParser(description='Train a Barlow Twins model')
    parser.add_argument('--fname', type=str, default=None,
                        help='Path to the project data file')
    parser.add_argument('--num_frames', type=str, default=None,
                        help='Path to the project data file')

    # Load the args
    args = parser.parse_args()
    fname = args.fname
    # Should be a pickle file
    with open(fname, 'rb') as f:
        model_args = pickle.load(f)

    # Load the model
    model_fname = model_args.model_fname
    state_dict = torch.load(model_fname)
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    target_sz = np.array(model_args.target_sz)
    backbone_kwargs = dict(in_channels=1, num_levels=2, f_maps=4, crop_sz=target_sz)
    model = BarlowTwins3d(model_args, backbone=ResidualEncoder3D, **backbone_kwargs).to(gpu)
    model.load_state_dict(state_dict)

    # Load the dataloader
    project_data1 = ProjectData.load_final_project_data_from_config(model_args.project_path)
    data_module = NeuronCropImageDataModule(project_data=project_data1,
                                            num_frames=args.num_frames,  # Not from the original args
                                            batch_size=1,
                                            train_fraction=1.0,
                                            crop_kwargs=dict(target_sz=target_sz))
    data_module.setup()
    loader = data_module.train_dataloader()

    # Evaluate
    model.eval()
    with torch.no_grad():
        for step, (y1, y2) in enumerate(loader):
            y1, y2 = torch.transpose(y1, 0, 1).type('torch.FloatTensor').to(gpu), torch.transpose(y2, 0, 1).type(
                'torch.FloatTensor').to(gpu)
            c, cT = model.calculate_both_correlation_matrices(y1, y2)
            visualize_model_performance(c)
            visualize_model_performance(cT)
            # break
