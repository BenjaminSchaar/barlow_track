# Template for barlow project training

## Folder Structure

The `barlow_project_template` directory is organized as follows:

```
barlow_project_template/
├── log/                                  # Output logs
├── checkpoints/                          # Output checkpoints
├── train_config.yaml                     # Configuration file for experiments
├── README.md                             # Project documentation
└── hyperparameter_search_template.txt    # Template for searching through hyperparameters, if used
```

## Training a network: basic

To start, you need:
1. A working conda environment
2. To clone this repository
3. A dataset in the Neurodata Without Borders file format
3b. Alternate file format: wbfm folder structure (recommended for now)

Copy the 'barlow_project_template' folder to the desired location then change the parameters in the config file as desired.
Next, you have two options: running it locally (you need a powerful computer) or on the cluster.

### Training on a cluster

```
conda activate wbfm
cd /path/to/this/repo
python barlow_track/scripts/sbatch_train_barlow_clusterer.sbatch -p /path/to/train_config.yaml
```

Important note: this will use a centrally located script to run the actual training (the .py file in the next section).
This means that if you need an update or otherwise the code is not working, the person maintaining that central code must update it.

### Training locally

```
conda activate wbfm
cd /path/to/this/repo
python barlow_track/scripts/train_barlow_clusterer.py -p /path/to/train_config.yaml
```

### Final output

This will produce output in the log/ and checkpoints/ folders, as well as the following in the main folder:
1. resnet50.pth - the final trained weights (this will be used when tracking)
2. args.pickle  - the hyperparameters for training; also needed for reading the network

If you are in the zimmer lab, you can add your trained network folder here:

'/lisc/scratch/neurobiology/zimmer/wbfm/TrainedBarlow'

## Optional: training multiple networks with hyperparameter search

From the barlow_project_template folder, make a parent folder with the hyperparameter_search_template.yaml file, and modify it to set which parameters to vary.
Optionally, set alternative defaults by also including a train_config.yaml file (this is required if you want to use wandb).
Then run:

```
conda activate wbfm
cd /path/to/this/repo
python barlow_track/scripts/optimize_hyperparameters.py -p /path/to/hyperparameter_search_template.yaml
```

By default this runs on the cluster, thus you may want to run it in a tmux session.
The python script will submit batch jobs (up to 100 total), and search through parameters to find the best loss.

This will create a series of nested folders in the parent folder, which are all copies of this single-project folder.
They are named trial_N, with the actually used hyperparameters saved in their individual train_config.yaml files.
Finally, the best parameters and trial are saved to "best_parameters.yaml" in the parent folder.
