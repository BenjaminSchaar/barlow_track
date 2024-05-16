# Load a project and data, then train a Siamese network
import logging
import os
import numpy as np
import sacred
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from wbfm.utils.projects.utils_filenames import get_sequential_filename
from wbfm.barlow_project.utils.data_loading import NeuronImageFeaturesDataModule
from wbfm.barlow_project.utils.model_image_classifier import NeuronEmbeddingModel
from wbfm.utils.projects.finished_project_data import ProjectData
import wandb
from sacred import Experiment

# !wandb login

ex = Experiment(save_git_info=False)
ex.add_config(project_path=None, max_epochs=30, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    project_data = ProjectData.load_final_project_data_from_config(project_path)


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)
    project_data = _config['project_data']

    seed = 43
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.info("Loading initial data...")

    ##
    logging.info("Loading ground truth annotations")
    df_manual_tracking = project_data.df_manual_tracking
    # neurons_that_are_finished = list(df_manual_tracking[df_manual_tracking['Finished?']]['Neuron ID'])
    # neurons_that_are_finished = list(df_manual_tracking[df_manual_tracking['auto-added tracklets correct']]['Neuron ID'])
    neurons_that_are_finished = list(df_manual_tracking[df_manual_tracking['first 1300 frames']]['Neuron ID'])
    num_finished = len(neurons_that_are_finished)

    print(f"Found {num_finished}/{len(df_manual_tracking)} finished neurons")

    ##
    logging.info("Initializing network and hyperparameters...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        gpus = 1
        print("Found cuda!")
    else:
        gpus = 0
        print("Did not find cuda")

    num_classes = num_finished
    batch_size = 2*1024
    max_epochs = _config['max_epochs']

    train_loader = NeuronImageFeaturesDataModule(batch_size=batch_size, project_data=project_data,
                                                 num_neurons=num_classes)
    model = NeuronEmbeddingModel(num_classes=num_classes)

    ##
    logging.info("Training network!")
    model.train()
    with wandb.init(project="basic_classifier_cluster", entity="charlesfieseler") as run:
        wandb_logger = WandbLogger()

        trainer = Trainer(gpus=gpus, max_epochs=max_epochs, terminate_on_nan=True,
                          stochastic_weight_avg=True,
                          logger=wandb_logger)
        wandb_logger.watch(model, log='all', log_freq=1)

        trainer.fit(model, train_loader)

    ## Save
    model_fname = f"classifier_{num_classes}_cluster.ckpt"
    training_folder = os.path.join(project_data.project_dir, 'nn_training')
    try:
        os.mkdir(training_folder)
    except OSError:
        pass
    model_fname = os.path.join(training_folder, model_fname)
    model_fname = get_sequential_filename(model_fname)
    trainer.save_checkpoint(model_fname)
