# Load a project and data, then train a Siamese network
import logging
import os
from pathlib import Path
import numpy as np
import sacred
import torch
from matplotlib import pyplot as plt

from wbfm.utils.nn_utils.utils_testing import test_trained_classifier, plot_accuracy, \
    test_trained_embedding_matcher
from wbfm.utils.projects.utils_filenames import add_name_suffix
from wbfm.barlow_project.utils.data_loading import NeuronImageFeaturesDataModule, FullVolumeNeuronImageFeaturesDataset
from wbfm.barlow_project.utils.model_image_classifier import NeuronEmbeddingModel
from wbfm.utils.projects.finished_project_data import ProjectData
from sacred import Experiment

# !wandb login

ex = Experiment(save_git_info=False)
ex.add_config(project_path=None, DEBUG=False)


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

    ##
    print("Loading ground truth annotations")
    df_manual_tracking = project_data.df_manual_tracking
    neurons_that_are_finished = list(df_manual_tracking[df_manual_tracking['first 1300 frames']]['Neuron ID'])
    num_finished = len(neurons_that_are_finished)

    print(f"Found {num_finished}/{len(df_manual_tracking)} finished neurons")

    ##
    print("Initializing network and hyperparameters...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        gpus = 1
        print("Found cuda!")
    else:
        gpus = 0
        print("Did not find cuda")

    num_classes = num_finished
    batch_size = 1024
    # First type of testing: as a classifier
    train_loader = NeuronImageFeaturesDataModule(batch_size=batch_size, project_data=project_data,
                                                 num_neurons=num_classes)
    train_loader.setup()
    # Second type of testing: as a tracker (matcher)
    full_loader = NeuronImageFeaturesDataModule(batch_size=batch_size, project_data=project_data,
                                                num_neurons=num_finished,
                                                base_dataset_class=FullVolumeNeuronImageFeaturesDataset)
    full_loader.setup()

    model_fname = f"classifier_{num_classes}_cluster.ckpt"
    training_folder = os.path.join(project_data.project_dir, 'nn_training')
    model_fname = os.path.join(training_folder, model_fname)
    # Enhancement: actually save the hyperparameters
    model = NeuronEmbeddingModel.load_from_checkpoint(model_fname, num_classes=num_classes)

    log_fname = str(Path(model_fname).with_suffix('.log'))
    log_fname = add_name_suffix(log_fname, '-log')
    logging.basicConfig(filename=log_fname)

    ##
    print("Calculating one-at-a-time classification accuracy")
    correct_per_class, total_per_class = test_trained_classifier(train_loader, model)
    plot_accuracy(correct_per_class, total_per_class)
    plt.xticks(rotation=90)
    plt.title("Accuracy as a one-at-a-time, time-independent classifier")

    fname = str(Path(model_fname).with_suffix('.png'))
    fname = add_name_suffix(fname, '-classifier_accuracy')
    plt.savefig(fname)

    ##
    print("Calculating tracker (matching) accuracy")
    correct_per_class, total_per_class = test_trained_embedding_matcher(full_loader, model)
    plot_accuracy(correct_per_class, total_per_class)
    plt.xticks(rotation=90)
    plt.title("Accuracy as a template-based matcher")

    fname = str(Path(model_fname).with_suffix('.png'))
    fname = add_name_suffix(fname, '-matcher_accuracy')
    plt.savefig(fname)
