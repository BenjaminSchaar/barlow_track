# Load a project and data, then train a Siamese network
import argparse
import json
import os
import pickle
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import sacred
import torch
import wandb
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import WandbLogger
from ruamel.yaml import YAML

from wbfm.barlow_project.utils.barlow import NeuronCropImageDataModule, BarlowTwins3d
from wbfm.barlow_project.utils.barlow_visualize import visualize_model_performance
from wbfm.barlow_project.utils.siamese import ResidualEncoder3D
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.projects.utils_filenames import get_sequential_filename


def main(args):
    # Set up logger
    wandb.login()

    # Load ground truth
    project_data1 = ProjectData.load_final_project_data_from_config(args.project_path)

    print("Preparing cropped volumes...")
    target_sz = np.array(args.target_sz)
    data_module = NeuronCropImageDataModule(project_data=project_data1, num_frames=args.num_frames, batch_size=1,
                                            train_fraction=args.train_fraction,
                                            val_fraction=args.val_fraction,
                                            crop_kwargs=dict(target_sz=target_sz))
    data_module.setup()
    loader = data_module.train_dataloader()

    torch.manual_seed(43)
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if gpu == "cpu":
        print("Initializing network using CPU...")
    else:
        print("Initializing network using GPU...")
    backbone_kwargs = dict(in_channels=1, num_levels=2, f_maps=4, crop_sz=target_sz)
    model = BarlowTwins3d(args, backbone=ResidualEncoder3D, **backbone_kwargs).to(gpu)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Actually train
    start_time = time.time()
    stats_file = get_sequential_filename(os.path.join(args.project_dir, 'log', 'stats.json'))
    checkpoint_file = get_sequential_filename(os.path.join(args.project_dir, 'checkpoints', 'checkpoint.pth'))
    print(f"Starting training. Stats in folder: {args.project_dir}")
    if args.dryrun:
        print("Dryrun, therefore stopping before actual training")
        return

    wandb_opt = dict(mode="disabled") if args.DEBUG else {}

    with wandb.init(project=args.wandb_name, entity="charlesfieseler", **wandb_opt) as run:
        # These lines are only useful for lightning projects
        # wandb_logger = WandbLogger()
        # wandb_logger.watch(model, log='all', log_freq=10)

        for epoch in range(0, args.epochs):
            for step, (y1, y2) in enumerate(loader, start=epoch * len(loader)):
                # Needs to be outside the data loader because the batch dimension isn't added yet
                y1, y2 = torch.transpose(y1, 0, 1).type('torch.FloatTensor'), torch.transpose(y2, 0, 1).type(
                    'torch.FloatTensor')
                y1 = y1.to(gpu)
                y2 = y2.to(gpu)

                # adjust_learning_rate(args, optimizer, loader, step)
                optimizer.zero_grad()
                loss, loss_original, loss_transpose = model.forward(y1, y2)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if step % args.print_freq == 0:
                    if args.rank == 0:
                        # Just print
                        stats = dict(epoch=epoch, step=step,
                                     loss=loss.item(),
                                     time=int(time.time() - start_time))
                        print(json.dumps(stats))
                        with open(stats_file, 'w') as f:
                            print(json.dumps(stats), file=f)

                        # wandb logging
                        run.log({"loss": loss, "loss_original": loss_original, "loss_transpose": loss_transpose})

                        # More infrequently, plot embedding
                        if step % (10*args.print_freq) == 0:
                            with torch.no_grad():
                                c = model.calculate_correlation_matrix(y1, y2)
                                save_fname = os.path.join(args.project_dir, 'log', f'correlation_matrix_{step}.png')
                                fig = visualize_model_performance(c, save_fname=save_fname)
                                run.log({"chart": fig})

            if args.rank == 0:
                # save checkpoint
                state = dict(epoch=epoch + 1, model=model.state_dict(),
                             optimizer=optimizer.state_dict())
                torch.save(state, checkpoint_file)

    # Final saving
    if args.rank == 0:
        # save final model (not in checkpoint dir)
        fname = get_sequential_filename(args.project_dir + '/resnet50.pth')
        torch.save(model.state_dict(), fname)
        args.model_fname = fname

    # Also save the args namespace
    fname = get_sequential_filename(args.project_dir + '/args.pickle')
    with open(fname, 'wb') as f:
        pickle.dump(args, f)


if __name__ == "__main__":
    # Get args, which is just path to yaml file
    parser = argparse.ArgumentParser(description='Train barlow network')
    parser.add_argument('--project_path', '-p', default=None,
                        help='path to yaml file (config)')

    cli_args = parser.parse_args()
    config_fname = cli_args.project_path

    # Load the yaml file
    with open(config_fname, 'r') as f:
        cfg = YAML().load(f)

    # Generate target saving locations from yaml location
    cfg['config_fname'] = config_fname
    cfg['project_dir'] = str(Path(config_fname).parent)
    args = SimpleNamespace(**cfg)
    # Run training code
    main(args)
