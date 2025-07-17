# Load a project and data, then train a Siamese network
import argparse
import json
import logging
import os
import pickle
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import wandb
from ruamel.yaml import YAML

from barlow_track.utils.barlow import BarlowTwins3d, load_barlow_model
from barlow_track.utils.barlow_lightning import NeuronCropImageDataModule
from barlow_track.utils.barlow_visualize import visualize_model_performance
from barlow_track.utils.siamese import ResidualEncoder3D

from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.general.utils_filenames import get_sequential_filename


def train_barlow_network(args):

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
    cuda_index = os.getenv("CUDA_VISIBLE_DEVICES", 0)
    gpu = torch.device(f"cuda:{cuda_index}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {gpu}")
    # Initialize model, loading from checkpoint if passed
    try:
        pretrained_model_path = args.pretrained_model_path
    except AttributeError:
        pretrained_model_path = None
    if pretrained_model_path is not None:
        logging.info(f"Loading model from {pretrained_model_path}")
        gpu, model, pretrained_args = load_barlow_model(pretrained_model_path)
        logging.info(f"Loaded pretrained args: {pretrained_args}")
        # Replace network-related values of args
        args.embedding_dim = pretrained_args.embedding_dim
        # Update hyperparameters with the user-passed new args
        for k, v in vars(args).items():
            setattr(model.args, k, v)
    else:
        user_args = vars(vars(args).get('backbone_kwargs', dict()))
        backbone_kwargs = dict(in_channels=1, num_levels=user_args.get('num_levels', 2), f_maps=user_args.get('f_maps', 4), crop_sz=target_sz)
        model = BarlowTwins3d(args, backbone=ResidualEncoder3D, **backbone_kwargs).to(gpu)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Actually train
    start_time = time.time()
    log_dir = os.path.join(args.project_dir, 'log')
    stats_file = get_sequential_filename(os.path.join(log_dir, 'stats.json'))
    checkpoint_file = get_sequential_filename(os.path.join(args.project_dir, 'checkpoints', 'checkpoint.pth'))
    print(f"Starting training with args: {args}. Stats in folder: {args.project_dir}")
    if args.dryrun:
        print("Dryrun, therefore stopping before actual training")
        return

    wandb_opt = dict(mode="disabled") if args.DEBUG else {}
    json_stats = []
    test_losses = dict(test_loss=0, test_loss_original=0, test_loss_transpose=0)

    # Initialize wandb run, if the user enables it
    if args.wandb_name and args.wandb_username:
        wandb.login()
        run = wandb.init(project=args.wandb_name, entity=args.wandb_username, **wandb_opt)
    else:
        run = None

    # Initial json entry: the wandb run name and id
    if run is not None:
        json_stats.append(dict(run_name=run.name, run_id=run.id))
    else:
        json_stats.append(dict(run_name="Non-wandb-run", run_id=None))

    try:
        for epoch in range(0, args.epochs):
            for step, (y1, y2) in enumerate(loader, start=epoch * len(loader)):
                y1, y2 = _format_vectors_on_gpu(y1, y2, gpu)

                # adjust_learning_rate(args, optimizer, loader, step)
                optimizer.zero_grad()
                loss, loss_original, loss_transpose = model.forward(y1, y2)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if step % args.print_freq == 0 or step == 0:
                    if args.rank == 0:
                        # Just print
                        stats = dict(epoch=epoch, step=step,
                                     loss=loss.item(), loss_original=loss_original.item(), loss_transpose=loss_transpose.item(),
                                     time=int(time.time() - start_time))
                        print(json.dumps(stats))
                        json_stats.append(stats)

                        # wandb logging
                        if run is not None:
                            run.log({"loss": loss.item(),
                                    "loss_original": loss_original.item(),
                                    "loss_transpose": loss_transpose.item()})

                        # More infrequently, plot embedding
                        if step % (10*args.print_freq) == 0:
                            with torch.no_grad():
                                c = model.calculate_correlation_matrix(y1, y2)
                                save_fname = os.path.join(args.project_dir, 'log', f'correlation_matrix_{step}.png')
                                fig = visualize_model_performance(c, save_fname=save_fname,
                                                                    vmin=-0.5, vmax=1)
                                if run is not None:
                                    run.log({"chart": fig})

            if args.rank == 0:
                # save checkpoint
                state = dict(epoch=epoch + 1, model=model.state_dict(),
                                optimizer=optimizer.state_dict())
                torch.save(state, checkpoint_file)
                # Calculate validation loss
                with torch.no_grad():
                    val_loss, val_loss_original, val_loss_transpose = 0, 0, 0
                    for val_step, (y1, y2) in enumerate(data_module.val_dataloader()):
                        y1, y2 = _format_vectors_on_gpu(y1, y2, gpu)
                        loss, loss_original, loss_transpose = model.forward(y1, y2)
                        val_loss += loss.item()
                        val_loss_original += loss_original.item()
                        val_loss_transpose += loss_transpose.item()
                        # Plot validation embedding
                        if val_step == 0 and run is not None:
                                c = model.calculate_correlation_matrix(y1, y2)
                                fig = visualize_model_performance(c, save_fname=None, vmin=-0.5, vmax=1)
                                run.log({"validation_chart": fig})

                if run is not None:
                    run.log({"val_loss": val_loss, "val_loss_original": val_loss_original, "val_loss_transpose": val_loss_transpose})
                # Printing
                stats = dict(epoch=epoch, val_loss=val_loss, time=int(time.time() - start_time))
                print(json.dumps(stats))
                json_stats.append(stats)

        # Calculate the final test loss
        test_loss, test_loss_original, test_loss_transpose = 0, 0, 0
        for test_step, (y1, y2) in enumerate(data_module.test_dataloader()):
            y1, y2 = _format_vectors_on_gpu(y1, y2, gpu)
            loss, loss_original, loss_transpose = model.forward(y1, y2)
            test_loss += loss.item()
            test_loss_original += loss_original.item()
            test_loss_transpose += loss_transpose.item()
        # Package losses into a dictionary for return
        test_losses = dict(test_loss=test_loss, test_loss_original=test_loss_original, test_loss_transpose=test_loss_transpose)

        if run is not None:
            run.log({"test_loss": test_loss, "test_loss_original": test_loss_original, "test_loss_transpose": test_loss_transpose})
        # Printing
        stats = dict(epoch=epoch, test_loss=test_loss, time=int(time.time() - start_time))
        print(json.dumps(stats))
        json_stats.append(stats)

    except KeyboardInterrupt:
        print("Interrupted training, saving model")
    except torch.cuda.OutOfMemoryError as e:
        print("Out of memory error, saving model")
        print(e)
    finally:
        # Clean up the wandb run
        if run is not None:
            run.finish()

        # Final saving
        with open(stats_file, 'w') as f:
            print(json.dumps(json_stats), file=f)

        if args.rank == 0:
            # save final model (not in checkpoint dir)
            fname = get_sequential_filename(args.project_dir + '/resnet50.pth')
            torch.save(model.state_dict(), fname)
            args.model_fname = fname

        # Also save the args namespace
        fname = get_sequential_filename(args.project_dir + '/args.pickle')
        with open(fname, 'wb') as f:
            pickle.dump(args, f)

        print("Training complete")
        
    return test_losses


def _format_vectors_on_gpu(y1, y2, gpu):
    # Needs to be outside the data loader because the batch dimension isn't added yet
    y1, y2 = torch.transpose(y1, 0, 1).type('torch.FloatTensor'), torch.transpose(y2, 0, 1).type(
        'torch.FloatTensor')
    y1 = y1.to(gpu)
    y2 = y2.to(gpu)
    return y1, y2


if __name__ == "__main__":
    # Get args, which is just path to yaml file
    parser = argparse.ArgumentParser(description='Train barlow network')
    parser.add_argument('--network_args', '-p', default=None,
                        help='path to yaml file (config)')

    cli_args = parser.parse_args()
    config_fname = cli_args.network_args

    # Load the yaml file
    with open(config_fname, 'r') as f:
        cfg = YAML().load(f)

    # Generate target saving locations from yaml location
    cfg['config_fname'] = config_fname
    cfg['project_dir'] = str(Path(config_fname).parent)
    args = SimpleNamespace(**cfg)
    # Run training code
    train_barlow_network(args)
